from collections import deque
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Dict
from page_cache_vector_env import GeneratorConfig, PageRequestGenerator

class FIFOCache:
    def __init__(self, capacity: int):
        assert capacity > 0
        self.capacity = capacity
        self.q = deque()
        self.set = set()

    def access(self, page: int) -> bool:
        # 命中
        if page in self.set:
            return True
        # 未命中：可能需要淘汰
        if len(self.q) >= self.capacity:
            ev = self.q.popleft()
            self.set.remove(ev)
        self.q.append(page)
        self.set.add(page)
        return False

class LRUCache:
    def __init__(self, capacity: int):
        assert capacity > 0
        self.capacity = capacity
        self.od = OrderedDict()

    def access(self, page: int) -> bool:
        hit = page in self.od
        if hit:
            # 访问后标为最近使用
            self.od.move_to_end(page, last=True)
            return True
        # 未命中：插入，必要时淘汰最久未使用
        if len(self.od) >= self.capacity:
            self.od.popitem(last=False)
        self.od[page] = None
        return False

class LFUCache:
    def __init__(self, capacity: int):
        assert capacity > 0
        self.capacity = capacity
        self.key2freq = {}                       # page -> freq
        self.freq2od = defaultdict(OrderedDict)  # freq -> OrderedDict(page -> None)，按 LRU 破平
        self.min_freq = 0
        self.size = 0

    def _touch(self, page: int, old_f: int):
        # 从旧频次桶移除
        od = self.freq2od[old_f]
        if page in od:
            od.pop(page)
        if not od and self.min_freq == old_f:
            self.min_freq += 1
        # 放入新频次桶
        new_f = old_f + 1
        self.freq2od[new_f][page] = None
        self.key2freq[page] = new_f

    def access(self, page: int) -> bool:
        # 命中：频次+1
        if page in self.key2freq:
            self._touch(page, self.key2freq[page])
            return True
        # 未命中：必要时淘汰
        if self.size >= self.capacity:
            # 淘汰 min\_freq 桶里最早进入的键（FIFO within freq）
            od = self.freq2od[self.min_freq]
            evict_page, _ = od.popitem(last=False)
            del self.key2freq[evict_page]
            if not od:
                del self.freq2od[self.min_freq]
        else:
            self.size += 1
        # 插入新页，freq=1，并更新 min\_freq
        self.freq2od[1][page] = None
        self.key2freq[page] = 1
        self.min_freq = 1
        return False

def gen_requests(cfg: GeneratorConfig, length: int, init_page: int = 0) -> List[int]:
    gen = PageRequestGenerator(cfg)
    gen.reset(init_page=init_page)
    seq = []
    for _ in range(length):
        seq.append(gen.next())
    return seq

def eval_cache(cache, requests: List[int]) -> Dict[str, float]:
    hits = 0
    for p in requests:
        if cache.access(p):
            hits += 1
    misses = len(requests) - hits
    hit_rate = hits / max(1, len(requests))
    return {"hits": hits, "misses": misses, "hit_rate": hit_rate}

def main():
    # 与 run.py 对齐的生成器配置
    N = 20
    capacity = 5
    req_len = 2000
    gen_cfg = GeneratorConfig(
        num_pages=N,
        p_repeat=0.2,
        p_local=0.5,
        local_window=3,
        seed=123,   # 固定种子以复现实验
    )

    # 生成同一请求序列（可选初始化页保证可复现）
    requests = gen_requests(gen_cfg, length=req_len, init_page=0)

    # 评测三种策略
    results = []
    for name, cls in [("FIFO", FIFOCache), ("LRU", LRUCache), ("LFU", LFUCache)]:
        cache = cls(capacity)
        res = eval_cache(cache, requests)
        results.append((name, res))

    # 打印结果
    print("=== Offline cache baselines on the same request sequence ===")
    for name, res in results:
        print(f"{name:4s} -> hit_rate={res['hit_rate']:.4f}  hits={res['hits']}  misses={res['misses']}")

if __name__ == "__main__":
    main()