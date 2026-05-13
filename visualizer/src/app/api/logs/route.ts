import { NextResponse } from 'next/server';
import { readdir, stat } from 'fs/promises';
import path from 'path';

export async function GET() {
  const logsDir = path.join(process.cwd(), 'public', 'logs');

  let entries;
  try {
    entries = await readdir(logsDir, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return NextResponse.json({ files: [] });
    }
    throw error;
  }

  const files = await Promise.all(
    entries
      .filter((entry) => entry.isFile() && entry.name.endsWith('.jsonl'))
      .map(async (entry) => {
        const filePath = path.join(logsDir, entry.name);
        const fileStat = await stat(filePath);
        return { name: entry.name, mtimeMs: fileStat.mtimeMs };
      })
  );

  files.sort((a, b) => b.mtimeMs - a.mtimeMs);

  return NextResponse.json({ files: files.slice(0, 10).map((file) => file.name) });
}
