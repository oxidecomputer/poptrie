# Poptrie

A poptrie is a data structure and set of algorithms for performing longest
prefix match of an IP address over a set of IP prefixes. Its primary use is
implementing routing tables.

Poptrie was created by Asai and Ohara in:

> Asai, Hirochika, and Yasuhiro Ohara. "Poptrie: A compressed trie with
> population count for fast and scalable software IP routing table lookup."
> ACM SIGCOMM Computer Communication Review 45.4 (2015): 57-70.

This is a dependency free `no_std` implementation to facilitate use in OS
kernels.

Performance analysis versus a naive LPM table is the following for a single
lookup. The naive lookup scales linearly and quickly exceeds the realm of
acceptable performance for data plane networking while the poptrie has near
constant scaling.

```
┌────────┬─────────┬────────┬─────────┐
│ Routes │ Poptrie │ Naive  │ Speedup │
├────────┼─────────┼────────┼─────────┤
│ 100    │ 5.9 ns  │ 426 ns │ 72x     │
├────────┼─────────┼────────┼─────────┤
│ 1,000  │ 5.9 ns  │ 4.9 µs │ 830x    │
├────────┼─────────┼────────┼─────────┤
│ 10,000 │ 10.3 ns │ 32 µs  │ 3,100x  │
└────────┴─────────┴────────┴─────────┘
````
