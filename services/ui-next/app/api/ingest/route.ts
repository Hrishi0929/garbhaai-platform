import { INGESTION_URL, forward } from "@/lib/proxy";

export async function POST(req: Request) {
  return forward(`${INGESTION_URL}/ingest`, req);
}
