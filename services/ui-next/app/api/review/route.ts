import { INFERENCE_URL, forward } from "@/lib/proxy";

export async function POST(req: Request) {
  return forward(`${INFERENCE_URL}/review`, req);
}
