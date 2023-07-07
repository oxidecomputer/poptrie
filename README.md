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
