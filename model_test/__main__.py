"""统一入口：python3 -m model_test <命令>（在 ModelBench 目录下运行）

  serve [--port N] [--open]   启动网页（默认命令；也可用 ./start.sh）
  run --endpoint 名称 ...      命令行跑一次测试（参数见 run --help）
  selftest                    仪器自检（本机假服务，验证平台自己测得准）
  test [unittest 参数]         单元测试
"""
import os
import sys
import unittest

USAGE = __doc__


def main(argv):
    cmd, rest = (argv[0], argv[1:]) if argv and not argv[0].startswith("-") else ("serve", argv)
    if cmd == "serve":
        from . import server
        sys.argv = ["python3 -m model_test serve"] + rest
        server.main()
    elif cmd == "run":
        from . import core
        sys.argv = ["python3 -m model_test run"] + rest
        core.main()
    elif cmd == "selftest":
        from . import selftest
        selftest.main()
    elif cmd == "test":
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        suite = unittest.defaultTestLoader.discover(os.path.join(root, "tests"), top_level_dir=root)
        sys.exit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    else:
        print(USAGE)
        sys.exit(0 if cmd in ("help", "-h", "--help") else 2)


if __name__ == "__main__":
    main(sys.argv[1:])
