import React, { useMemo, useState } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  flexRender,
  ColumnDef,
  SortingState,
} from '@tanstack/react-table';
import { Trash2, Download, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';
import type { GameReplay } from '../../types';

interface GameReplayTableProps {
  games: GameReplay[];
  selectedGame: GameReplay | null;
  onSelect: (game: GameReplay) => void;
  onDelete: (index: number, event: React.MouseEvent) => void;
  onDeleteAll: () => void;
  onExportAll: () => void;
}

type Row = GameReplay & { index: number };

function SortHeader({
  column,
  label,
}: {
  column: { toggleSorting: (desc: boolean) => void; getIsSorted: () => string | false };
  label: React.ReactNode;
}) {
  const sorted = column.getIsSorted();
  return (
    <button
      className="flex items-center gap-1 hover:bg-gray-100 dark:hover:bg-gray-700 px-2 py-1 rounded"
      onClick={() => column.toggleSorting(sorted === 'asc')}
    >
      {label}
      {sorted === 'asc' ? (
        <ArrowUp className="w-4 h-4" />
      ) : sorted === 'desc' ? (
        <ArrowDown className="w-4 h-4" />
      ) : (
        <ArrowUpDown className="w-4 h-4" />
      )}
    </button>
  );
}

const GameReplayTable: React.FC<GameReplayTableProps> = ({
  games,
  selectedGame,
  onSelect,
  onDelete,
  onDeleteAll,
  onExportAll,
}) => {
  const [sorting, setSorting] = useState<SortingState>([{ id: 'timestamp', desc: true }]);

  const columns = useMemo<ColumnDef<Row>[]>(
    () => [
      {
        accessorKey: 'teamName',
        header: ({ column }) => <SortHeader column={column} label="Team" />,
        cell: ({ row }) => (
          <div className="font-medium text-gray-900 dark:text-gray-100">
            {row.original.teamName || 'Unknown Team'}
          </div>
        ),
      },
      {
        accessorKey: 'scenarioName',
        header: ({ column }) => <SortHeader column={column} label="Scenario" />,
        cell: ({ row }) => (
          <div className="text-gray-600 dark:text-gray-400">
            {row.original.scenarioName || 'Custom Scenario'}
          </div>
        ),
      },
      {
        accessorKey: 'timestamp',
        header: ({ column }) => <SortHeader column={column} label="Date" />,
        cell: ({ row }) => {
          const timestamp = row.original.timestamp ? new Date(row.original.timestamp) : null;
          return (
            <div className="text-sm text-gray-500 dark:text-gray-500">
              {timestamp ? timestamp.toLocaleString() : 'No timestamp'}
            </div>
          );
        },
      },
      {
        accessorKey: 'nights',
        header: ({ column }) => <SortHeader column={column} label="Nights" />,
        cell: ({ row }) => (
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {row.original.nights.length}
          </div>
        ),
      },
      {
        accessorKey: 'num_agents',
        header: ({ column }) => <SortHeader column={column} label="Agents" />,
        cell: ({ row }) => (
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {row.original.scenario.num_agents}
          </div>
        ),
      },
      {
        accessorKey: 'beta',
        header: ({ column }) => <SortHeader column={column} label="β" />,
        cell: ({ row }) => (
          <div className="text-sm font-mono text-gray-600 dark:text-gray-400">
            {row.original.scenario.beta.toFixed(2)}
          </div>
        ),
      },
      {
        accessorKey: 'kappa',
        header: ({ column }) => <SortHeader column={column} label="κ" />,
        cell: ({ row }) => (
          <div className="text-sm font-mono text-gray-600 dark:text-gray-400">
            {row.original.scenario.kappa.toFixed(2)}
          </div>
        ),
      },
      {
        id: 'actions',
        header: 'Actions',
        cell: ({ row }) => (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(row.original.index, e);
            }}
            className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
            title="Delete this replay"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        ),
      },
    ],
    [onDelete]
  );

  const tableData = useMemo(
    () => games.map((game, index) => ({ ...game, index })),
    [games]
  );

  const table = useReactTable({
    data: tableData,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onSortingChange: setSorting,
    state: {
      sorting,
    },
    initialState: {
      pagination: {
        pageSize: 10,
      },
    },
  });

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          Game Replays ({games.length})
        </h2>
        <div className="flex gap-2">
          <button
            onClick={onExportAll}
            disabled={games.length === 0}
            className="btn-secondary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Export all games as JSON"
          >
            <Download className="w-4 h-4" />
            Export All
          </button>
          <button
            onClick={onDeleteAll}
            disabled={games.length === 0}
            className="btn-secondary flex items-center gap-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Delete all game replays"
          >
            <Trash2 className="w-4 h-4" />
            Delete All
          </button>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-800">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                  >
                    {header.isPlaceholder
                      ? null
                      : flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                onClick={() => onSelect(row.original)}
                className={`cursor-pointer transition-colors ${
                  selectedGame === row.original
                    ? 'bg-blue-50 dark:bg-blue-950 border-l-4 border-blue-500'
                    : 'hover:bg-gray-50 dark:hover:bg-gray-800'
                }`}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-6 py-4 whitespace-nowrap text-sm">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>

        {/* Pagination Controls */}
        {games.length > 10 && (
          <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-700 dark:text-gray-300">
              Showing {table.getState().pagination.pageIndex * table.getState().pagination.pageSize + 1} to{' '}
              {Math.min(
                (table.getState().pagination.pageIndex + 1) * table.getState().pagination.pageSize,
                games.length
              )}{' '}
              of {games.length} games
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => table.previousPage()}
                disabled={!table.getCanPreviousPage()}
                className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
              </span>
              <button
                onClick={() => table.nextPage()}
                disabled={!table.getCanNextPage()}
                className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GameReplayTable;
