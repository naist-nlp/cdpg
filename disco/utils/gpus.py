import socket
from gpustat.core import GPUStatCollection
import sys

def get_device_name():
    # コンピュータ名を取得
    host = socket.gethostname()
    #print(host)
    return host


def query_gpus():
    try:
        return GPUStatCollection.new_query()
    except Exception as e:
        print("Error on querying GPUs.", file=sys.stderr)
        sys.exit(1)


def get_idle_gpus(num_gpus=2, quiet=True):
    if num_gpus == 0:
        return []

    idles_gpus = [str(p["index"]) for p in query_gpus() if len(p["processes"]) == 0]
    if len(idles_gpus) < num_gpus:
        if not quiet:
            print(
                "There are no available GPUs. (# of GPU querys: {})".format(num_gpus),
                file=sys.stderr,
            )
        return None
    return idles_gpus[:num_gpus]
