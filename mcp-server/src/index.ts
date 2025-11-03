#!/usr/bin/env node

/**
 * MCP Remote SSH Server
 *
 * Provides tools for executing commands on remote machines via SSH.
 * This allows Claude to directly interact with remote development/GPU machines.
 */

import { config as loadEnv } from "dotenv";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { Client } from "ssh2";

// Load .env file from project root (two directories up from dist/)
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const envPath = resolve(__dirname, "../../.env");
loadEnv({ path: envPath });

interface SSHConfig {
  host: string;
  port: number;
  username: string;
  privateKeyPath?: string;
}

interface BackgroundJob {
  id: string;
  command: string;
  stream: any;
  output: string[];
  exitCode?: number;
}

class RemoteSSHServer {
  private server: Server;
  private sshClient: Client | null = null;
  private config: SSHConfig;
  private backgroundJobs = new Map<string, BackgroundJob>();
  private jobCounter = 0;

  constructor() {
    // Get SSH config from environment
    this.config = this.loadConfig();

    this.server = new Server(
      {
        name: "remote-ssh",
        version: "0.1.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  private loadConfig(): SSHConfig {
    const sshHost = process.env.SSH_HOST || "";
    const [username, host] = sshHost.includes("@")
      ? sshHost.split("@")
      : ["", sshHost];

    return {
      host: host || "localhost",
      port: parseInt(process.env.SSH_PORT || "22"),
      username: username || process.env.USER || "root",
      privateKeyPath: process.env.SSH_KEY_PATH,
    };
  }

  private async ensureConnection(): Promise<Client> {
    if (this.sshClient && (this.sshClient as any)._sock?.readable) {
      return this.sshClient;
    }

    return new Promise((resolve, reject) => {
      const client = new Client();

      client.on("ready", () => {
        this.sshClient = client;
        console.error(`✓ SSH connected to ${this.config.username}@${this.config.host}`);
        resolve(client);
      });

      client.on("error", (err) => {
        console.error(`✗ SSH connection error: ${err.message}`);
        reject(err);
      });

      const connectConfig: any = {
        host: this.config.host,
        port: this.config.port,
        username: this.config.username,
      };

      if (this.config.privateKeyPath) {
        const fs = require("fs");
        connectConfig.privateKey = fs.readFileSync(this.config.privateKeyPath);
      }

      client.connect(connectConfig);
    });
  }

  private setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: this.getTools(),
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      return await this.handleToolCall(request);
    });
  }

  private getTools(): Tool[] {
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

  private async handleToolCall(request: any): Promise<{
    content: Array<{ type: string; text: string }>;
    isError?: boolean;
  }> {
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
    } catch (error: any) {
      return {
        content: [
          {
            type: "text" as const,
            text: `Error: ${error.message}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async executeBash(args: any): Promise<{
    content: Array<{ type: string; text: string }>;
  }> {
    const { command, description, run_in_background, timeout = 120000 } = args;

    const client = await this.ensureConnection();

    if (run_in_background) {
      return this.executeBashBackground(client, command);
    }

    return new Promise((resolve, reject) => {
      let output = "";
      let errorOutput = "";

      client.exec(command, (err, stream) => {
        if (err) {
          reject(err);
          return;
        }

        const timeoutId = setTimeout(() => {
          stream.close();
          reject(new Error("Command timed out"));
        }, timeout);

        stream.on("close", (code: number) => {
          clearTimeout(timeoutId);
          resolve({
            content: [
              {
                type: "text",
                text: output || errorOutput || "(no output)",
              },
            ],
          });
        });

        stream.on("data", (data: Buffer) => {
          output += data.toString();
        });

        stream.stderr.on("data", (data: Buffer) => {
          errorOutput += data.toString();
        });
      });
    });
  }

  private async executeBashBackground(client: Client, command: string): Promise<{
    content: Array<{ type: string; text: string }>;
  }> {
    const jobId = `remote-${++this.jobCounter}`;
    const job: BackgroundJob = {
      id: jobId,
      command,
      stream: null,
      output: [],
    };

    this.backgroundJobs.set(jobId, job);

    client.exec(command, (err, stream) => {
      if (err) {
        job.output.push(`Error: ${err.message}`);
        job.exitCode = 1;
        return;
      }

      job.stream = stream;

      stream.on("close", (code: number) => {
        job.exitCode = code;
      });

      stream.on("data", (data: Buffer) => {
        job.output.push(data.toString());
      });

      stream.stderr.on("data", (data: Buffer) => {
        job.output.push(data.toString());
      });
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

  private async getBashOutput(args: any): Promise<{
    content: Array<{ type: string; text: string }>;
  }> {
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

  private async readFile(args: any): Promise<{
    content: Array<{ type: string; text: string }>;
  }> {
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
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("MCP Remote SSH Server running on stdio");
    console.error(`Target: ${this.config.username}@${this.config.host}:${this.config.port}`);
  }
}

// Start server
const server = new RemoteSSHServer();
server.run().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
