"""demo.py - 30-second live demo of the Min-Heap triage line (good for the viva).  python demo.py"""
from min_heap import MinHeap

SEV = {1: "RESUSCITATION", 2: "EMERGENT", 3: "URGENT", 4: "LESS URGENT", 5: "NON-URGENT"}
arrivals = [("Ali (sprained wrist)", 5), ("Sara (fever)", 4), ("Bilal (chest pain)", 2),
            ("Hina (cut finger)", 5), ("Omar (cardiac arrest)", 1), ("Zoya (broken leg)", 3),
            ("Taha (severe bleeding)", 2)]
q = MinHeap()
for t, (name, sev) in enumerate(arrivals):
    q.push((sev, t), name)                 # key = (severity, arrival order)
    print(f"t={t}  arrives: {name:<24} severity {sev} ({SEV[sev]})   queue size = {len(q)}")
print("\n--- doctor calls patients in this order ---")
while len(q):
    (sev, t), name = q.pop()
    print(f"severity {sev}  (arrived #{t})  ->  {name}")
