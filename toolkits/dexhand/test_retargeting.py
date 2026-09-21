# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""独立验证手套重定向与 RViz 显示；不连接真机。"""

import argparse
import time

from rlinf_dexhand.pipeline import TeleopPipeline, load_config

from toolkits.dexhand.transport import RvizClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--frequency", type=float, default=30)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.frequency <= 0 or (args.seconds is not None and args.seconds <= 0):
        parser.error("频率和时长必须为正数")
    cfg = load_config(args.config)
    if cfg["hand"]["type"] != "wuji1hand":
        parser.error("此 RViz 测试仅用于 wuji1hand")
    pipeline = TeleopPipeline(cfg)
    client = RvizClient(pipeline.spec, args.endpoint)
    start = time.monotonic()
    count = 0
    try:
        client.start()
        pipeline.start()
        while args.seconds is None or time.monotonic() - start < args.seconds:
            tick = time.monotonic()
            target = pipeline.read()
            client.send(target)
            count += 1
            time.sleep(max(0, 1 / args.frequency - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pipeline.close()
        finally:
            client.close()
        elapsed = time.monotonic() - start
        print(
            f"测试结束：{count} 帧，{elapsed:.2f} 秒，{count / max(elapsed, 1e-9):.2f} Hz"
        )


if __name__ == "__main__":
    main()
