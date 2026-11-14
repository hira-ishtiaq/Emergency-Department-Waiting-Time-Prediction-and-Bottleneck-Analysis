"""
min_heap.py - Binary Min-Heap implemented from scratch (no heapq).

This is the core Data Structure of PulseMetrics. Every waiting line in the
emergency department (triage, doctor, lab) AND the simulation's event
calendar are Min-Heaps.

Why a heap?
    * push      : O(log n)   (sift-up)
    * pop-min   : O(log n)   (sift-down)
    * peek-min  : O(1)
  A sorted list would need O(n) per insert, and an unsorted list O(n) per
  "who is next?" lookup. A heap gives both in logarithmic time.

Storage: an array where the children of index i are 2i+1 and 2i+2 and the
parent of i is (i-1)//2.

Stable tie-breaking: each entry is (key, counter, item). The counter always
increases, so two entries with the same key come out in insertion order
(first-come-first-served inside the same priority level) and the `item`
itself is never compared.
"""


class MinHeap:
    def __init__(self):
        self._a = []          # list of (key, counter, item)
        self._counter = 0

    # ---------------------------------------------------------- basic API
    def __len__(self):
        return len(self._a)

    def is_empty(self):
        return not self._a

    def push(self, key, item):
        """Insert `item` with priority `key` (smaller key = served earlier). O(log n)."""
        self._a.append((key, self._counter, item))
        self._counter += 1
        self._sift_up(len(self._a) - 1)

    def peek(self):
        """Return (key, item) of the minimum without removing it. O(1)."""
        if not self._a:
            raise IndexError("peek from empty heap")
        key, _, item = self._a[0]
        return key, item

    def pop(self):
        """Remove and return (key, item) with the smallest key. O(log n)."""
        if not self._a:
            raise IndexError("pop from empty heap")
        a = self._a
        top = a[0]
        last = a.pop()
        if a:                       # move last leaf to the root and restore order
            a[0] = last
            self._sift_down(0)
        return top[0], top[2]

    # ------------------------------------------------------------ internals
    def _sift_up(self, i):
        a = self._a
        while i > 0:
            parent = (i - 1) // 2
            if a[i] < a[parent]:    # tuple compare: key first, then counter
                a[i], a[parent] = a[parent], a[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        a = self._a
        n = len(a)
        while True:
            left, right, smallest = 2 * i + 1, 2 * i + 2, i
            if left < n and a[left] < a[smallest]:
                smallest = left
            if right < n and a[right] < a[smallest]:
                smallest = right
            if smallest == i:
                break
            a[i], a[smallest] = a[smallest], a[i]
            i = smallest

    def items_sorted(self):
        """Non-destructive view of all (key, item) in service order (used by the dashboard)."""
        return [(k, i) for k, _, i in sorted(self._a, key=lambda e: (e[0], e[1]))]

    def is_valid_heap(self):
        """Check the heap property (used by the unit tests)."""
        a = self._a
        return all(a[(i - 1) // 2] <= a[i] for i in range(1, len(a)))
