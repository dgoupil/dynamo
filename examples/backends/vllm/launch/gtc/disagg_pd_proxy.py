#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
disagg_pd_proxy.py - Proxy for Prefill/Decode disaggregation.

Routes requests through the two-stage PD pipeline:
  1. Forward request to a prefill instance (round-robin).
     The prefill instance handles encoding + prefill and pushes KV via NIXL.
  2. Forward the original request (with kv_transfer_params from prefill)
     to the decode instance, which pulls KV and generates tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import uuid
from collections.abc import AsyncIterator

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("pd-proxy")

app = FastAPI()
prefill_session = None
decode_session = None


@app.on_event("startup")
async def on_startup():
    global prefill_session, decode_session
    timeout = aiohttp.ClientTimeout(total=100_000)
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    prefill_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    decode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)


@app.on_event("shutdown")
async def on_shutdown():
    if prefill_session:
        await prefill_session.close()
    if decode_session:
        await decode_session.close()


async def do_prefill(req_data: dict, p_url: str, req_id: str) -> dict:
    """Send request to prefill instance, return kv_transfer_params."""
    prefill_req = req_data.copy()
    prefill_req["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    prefill_req["stream"] = False
    prefill_req["max_tokens"] = 1
    if "max_completion_tokens" in prefill_req:
        prefill_req["max_completion_tokens"] = 1
    prefill_req.pop("stream_options", None)

    headers = {"x-request-id": req_id}
    resp = await prefill_session.post(
        f"{p_url}/v1/chat/completions", json=prefill_req, headers=headers
    )
    if resp.status != 200:
        detail = await resp.text()
        logger.error("[%s] Prefill failed (%s): %s", req_id, resp.status, detail)
        raise HTTPException(status_code=resp.status, detail=detail)

    body = await resp.json()
    kv_params = body.get("kv_transfer_params", {})
    if kv_params:
        req_data["kv_transfer_params"] = kv_params
    logger.info("[%s] Prefill done on %s", req_id, p_url)
    return req_data


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        req_data = await request.json()
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        p_url = random.choice(app.state.p_urls)
        d_url = random.choice(app.state.d_urls)

        # Step 1: prefill (encoding + prefill + KV push)
        req_data = await do_prefill(req_data, p_url, req_id)

        # Step 2: decode
        headers = {"x-request-id": req_id}
        is_streaming = req_data.get("stream", False)

        if is_streaming:
            async def stream_gen():
                async with decode_session.post(
                    f"{d_url}/v1/chat/completions",
                    json=req_data,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.content.iter_chunked(1024):
                        if chunk:
                            yield chunk.decode("utf-8", errors="ignore")

            return StreamingResponse(stream_gen(), media_type="text/event-stream")

        async with decode_session.post(
            f"{d_url}/v1/chat/completions", json=req_data, headers=headers
        ) as resp:
            resp.raise_for_status()
            return JSONResponse(content=await resp.json())

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] Error: %s", req_id, str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/models")
async def list_models():
    async with decode_session.get(f"{app.state.d_urls[0]}/v1/models") as resp:
        resp.raise_for_status()
        return await resp.json()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "prefill_servers": len(app.state.p_urls),
        "decode_servers": len(app.state.d_urls),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PD disaggregation proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--prefill-servers-urls", required=True,
        help="Comma-separated prefill server URLs",
    )
    parser.add_argument(
        "--decode-servers-urls", required=True,
        help="Comma-separated decode server URLs",
    )
    args = parser.parse_args()

    app.state.p_urls = [u.strip() for u in args.prefill_servers_urls.split(",") if u.strip()]
    app.state.d_urls = [u.strip() for u in args.decode_servers_urls.split(",") if u.strip()]

    logger.info("PD Proxy on %s:%s", args.host, args.port)
    logger.info("  Prefill: %s", app.state.p_urls)
    logger.info("  Decode:  %s", app.state.d_urls)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
