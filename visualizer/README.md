# RLM Visualizer

Vibe-coded visualizer with [shadcn](https://ui.shadcn.com) for viewing RLM trajectories.

## Getting Started

Run commands from this directory:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Visualize a Local Rollout Log

From the repository root, copy the JSONL rollout into the visualizer's public
logs directory:

```bash
mkdir -p visualizer/public/logs
cp logs/rlm_2026-05-12_15-45-25_2ce0632b.jsonl visualizer/public/logs/
```

Start the visualizer from `visualizer/`. Use another port if `3000` is already
busy:

```bash
cd visualizer
npm run dev -- -p 3000
```

Open the log directly with the `log` query parameter:

```text
http://localhost:3000/?log=rlm_2026-05-12_15-45-25_2ce0632b.jsonl
```

You can also open `http://localhost:3000` and select the file from the
Recent Traces list. The app reads that list from `public/logs/*.jsonl`.

## Verify Changes

Before relying on visualizer changes, run:

```bash
npm run lint
npm run build
```

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
