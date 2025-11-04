#!/usr/bin/env node
/**
 * MCP Remote SSH Server
 *
 * Provides tools for executing commands on remote machines via SSH.
 * This allows Claude to directly interact with remote development/GPU machines.
 *
 * Uses native SSH command to support full SSH config including ProxyCommand,
 * ControlMaster, and other advanced SSH features.
 */
import { config as loadEnv } from "dotenv";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
// Load .env file from project root (two directories up from dist/)
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const envPath = resolve(__dirname, "../../.env");
loadEnv({ path: envPath });
class RemoteSSHServer {
    server;
    config;
    backgroundJobs = new Map();
    jobCounter = 0;
    constructor() {
        // Get SSH config from environment
        this.config = this.loadConfig();
        this.server = new Server({
            name: "remote-ssh",
            version: "0.2.0",
        }, {
            capabilities: {
                tools: {},
            },
        });
        this.setupHandlers();
    }
    loadConfig() {
        // SSH_HOST should now be an SSH config alias or user@host format
        const sshHost = process.env.SSH_HOST || "localhost";
        return {
            host: sshHost,
        };
    }
    async testConnection() {
        return new Promise((resolve) => {
            const proc = spawn("ssh", [this.config.host, "echo", "test"]);
            let success = false;
            proc.on("exit", (code) => {
                success = code === 0;
                resolve(success);
            });
            proc.on("error", () => {
                resolve(false);
            });
            // Timeout after 10 seconds
            setTimeout(() => {
                if (!success) {
                    proc.kill();
                    resolve(false);
                }
            }, 10000);
        });
    }
    setupHandlers() {
        this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
            tools: this.getTools(),
        }));
        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            return await this.handleToolCall(request);
        });
    }
    getTools() {
        return [
            {
                name: "remote_bash",
                description: "Execute a bash command on the remote SSH host",
                inputSchema: {
                    type: "object",
                    properties: {
                        command: {
                            type: "string",
                            description: "The bash command to execute",
                        },
                        description: {
                            type: "string",
                            description: "Human-readable description of what the command does",
                        },
                        run_in_background: {
                            type: "boolean",
                            description: "Run command in background and return job ID",
                            default: false,
                        },
                        timeout: {
                            type: "number",
                            description: "Timeout in milliseconds (max 600000)",
                            default: 120000,
                        },
                    },
                    required: ["command"],
                },
            },
            {
                name: "remote_bash_output",
                description: "Get output from a background remote bash job",
                inputSchema: {
                    type: "object",
                    properties: {
                        bash_id: {
                            type: "string",
                            description: "The ID of the background job",
                        },
                    },
                    required: ["bash_id"],
                },
            },
            {
                name: "remote_file_read",
                description: "Read a file from the remote host",
                inputSchema: {
                    type: "object",
                    properties: {
                        file_path: {
                            type: "string",
                            description: "Absolute path to the file on remote host",
                        },
                        offset: {
                            type: "number",
                            description: "Line number to start reading from",
                        },
                        limit: {
                            type: "number",
                            description: "Number of lines to read",
                        },
                    },
                    required: ["file_path"],
                },
            },
        ];
    }
    async handleToolCall(request) {
        const { name, arguments: args } = request.params;
        try {
            switch (name) {
                case "remote_bash":
                    return await this.executeBash(args);
                case "remote_bash_output":
                    return await this.getBashOutput(args);
                case "remote_file_read":
                    return await this.readFile(args);
                default:
                    throw new Error(`Unknown tool: ${name}`);
            }
        }
        catch (error) {
            return {
                content: [
                    {
                        type: "text",
                        text: `Error: ${error.message}`,
                    },
                ],
                isError: true,
            };
        }
    }
    async executeBash(args) {
        const { command, description, run_in_background, timeout = 120000 } = args;
        if (run_in_background) {
            return this.executeBashBackground(command);
        }
        return new Promise((resolve, reject) => {
            let output = "";
            let errorOutput = "";
            // Use SSH command to execute remote command
            // SSH will handle all config including ProxyCommand
            const proc = spawn("ssh", [this.config.host, command]);
            const timeoutId = setTimeout(() => {
                proc.kill();
                reject(new Error("Command timed out"));
            }, timeout);
            proc.stdout.on("data", (data) => {
                output += data.toString();
            });
            proc.stderr.on("data", (data) => {
                errorOutput += data.toString();
            });
            proc.on("close", (code) => {
                clearTimeout(timeoutId);
                // Combine stdout and stderr, prefer stdout if available
                const result = output || errorOutput || "(no output)";
                resolve({
                    content: [
                        {
                            type: "text",
                            text: result,
                        },
                    ],
                });
            });
            proc.on("error", (err) => {
                clearTimeout(timeoutId);
                reject(err);
            });
        });
    }
    async executeBashBackground(command) {
        const jobId = `remote-${++this.jobCounter}`;
        // Use SSH command in background
        const proc = spawn("ssh", [this.config.host, command]);
        const job = {
            id: jobId,
            command,
            process: proc,
            output: [],
        };
        this.backgroundJobs.set(jobId, job);
        proc.stdout.on("data", (data) => {
            job.output.push(data.toString());
        });
        proc.stderr.on("data", (data) => {
            job.output.push(data.toString());
        });
        proc.on("close", (code) => {
            job.exitCode = code;
        });
        proc.on("error", (err) => {
            job.output.push(`Error: ${err.message}`);
            job.exitCode = 1;
        });
        return {
            content: [
                {
                    type: "text",
                    text: `Command running in background with ID: ${jobId}`,
                },
            ],
        };
    }
    async getBashOutput(args) {
        const { bash_id } = args;
        const job = this.backgroundJobs.get(bash_id);
        if (!job) {
            throw new Error(`Job ${bash_id} not found`);
        }
        const output = job.output.join("");
        const status = job.exitCode !== undefined ? "completed" : "running";
        return {
            content: [
                {
                    type: "text",
                    text: `Status: ${status}\nExit code: ${job.exitCode ?? "N/A"}\n\n${output}`,
                },
            ],
        };
    }
    async readFile(args) {
        const { file_path, offset, limit } = args;
        let command = `cat "${file_path}"`;
        if (offset !== undefined || limit !== undefined) {
            const tailCmd = offset ? `tail -n +${offset}` : "cat";
            const headCmd = limit ? `head -n ${limit}` : "cat";
            command = `${tailCmd} "${file_path}" | ${headCmd}`;
        }
        // Add line numbers
        command += " | cat -n";
        return this.executeBash({ command });
    }
    async run() {
        // Test connection on startup
        console.error("MCP Remote SSH Server starting...");
        console.error(`Target: ${this.config.host}`);
        console.error("Testing connection...");
        const connected = await this.testConnection();
        if (connected) {
            console.error("✓ SSH connection test successful");
        }
        else {
            console.error("✗ SSH connection test failed - commands may fail");
            console.error("  Check that SSH_HOST is configured correctly in .env");
        }
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        console.error("MCP Remote SSH Server running on stdio");
    }
}
// Start server
const server = new RemoteSSHServer();
server.run().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
});
//# sourceMappingURL=index.js.map