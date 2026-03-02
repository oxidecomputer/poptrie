// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Copyright 2026 Oxide Computer Company

#![no_std]

//! A poptrie implementation.
//!
//! A poptrie is a data structure and set of algorithms for performing longest
//! prefix match of an IP address over a set of IP prefixes. Its primary use is
//! implementing routing tables.
//!
//! Poptrie was created by Asai and Ohara in:
//!
//! > Asai, Hirochika, and Yasuhiro Ohara. "Poptrie: A compressed trie with
//! > population count for fast and scalable software IP routing table lookup."
//! > ACM SIGCOMM Computer Communication Review 45.4 (2015): 57-70.
//!
//! This is a dependency free `no_std` implementation to facilitate use in OS
//! kernels.
//!
//! ## By Example
//!
//! This section describes describes a poptrie by building and querying one.
//! Consider a routing table with the following entries.
//!
//! ```text
//! Destination     Nexthop
//! -----------     -------
//! 1.0.0.0/8       1.254.254.254
//! 247.33.0.0/16   247.33.0.1
//! 247.33.12.0/24  247.33.12.1
//! 51.12.109.0/24  51.12.109.10
//! 77.18.0.0/16    77.18.10.1
//! 170.1.14.3/32   1.7.0.1
//! 0.0.0.0.0/0     1.2.3.4
//! ```
//! We are going to build a poptrie based on 64-bit bitmaps. The way that
//! poptrie works is by breaking up the IP address we need a nexthop for into
//! prefix chunks. The nodes of a poptrie contain bitmaps. Each one in the
//! bitmap is a pointer to a child node. This means there up to 64 child nodes
//! for any given poptrie node. Therefore if a prefix chunk needs to map onto
//! one of 64 values, the prefix may only contain up to 64 values. This means
//! the prefixes must have a numeric value no larger than 64 which is another
//! way of saying they can be no larger than 6 bits.
//!
//! Let's start to construct a poptrie for the routing table described above.
//!
//! ```text
//! Destination     Numeric value of fist 6 bits
//! -----------     ----------------------------
//! 1.0.0.0/8       0
//! 247.33.0.0/16   61
//! 247.33.12.0/24  61
//! 51.12.109.0/24  12
//! 77.18.0.0/16    19
//! 170.1.14.3/32   42
//! 0.0.0.0.0/0     .
//! ```
//!
//! This gives us a root poptrie node that looks like this.
//!
//! ```text
//!     6    4    1    1
//!     1    2    9    2    0  
//! +-------------------------+
//! |..|1|..|1|..|1|..|1|..|1||
//! +-------------------------+
//! ```
//!
//! The first five destinations in the routing table have prefix lengths that
//! are greater than 6. Therefore, a `1` is placed at the bitmap location
//! corresponding to the numeric value of the first 6 bits of the prefix,
//! indicating that further tree traversal is required in order to match those
//! prefixes. The last entry in the routing table, the default route, has a
//! prefix length less than 6. This means the first 6 bits of any query is
//! sufficient to match against this prefix and a `0` is placed at the
//! corresponding bitmap location.
//!
//! We'll refer to this bitmap as `v`.
//!
//! For any IP address, if we call the numeric value of the first 6 bits `n`, we
//! have a selector `sel` as follows.
//!
//! ```text
//! sel = 1 << n
//! ```
//!
//! The bitwise intersection of `sel` and `v` tells us what kind of node is next
//! in the trie. If `sel & v` is zero, then we have reached a leaf node.
//! Otherwise we have reached an interior node. The root of the tree is always
//! an interior node. The poptrie data structure keeps interior and leaf nodes
//! in two distinct arrays.
//!
//! ```
//! use poptrie::*;
//! pub struct Poptrie<T> {
//!     pub interior: Vec<Interior>,
//!     pub leaf: Vec<Leaf<T>>,
//!     pub default: Option<Leaf<T>>,
//! }
//! ```
//!
//! Once we have determined if a selector leads us to an interior or leaf node,
//! we need to combine `sel` and `v` to form an index into the corresponding
//! array. In the case of an interior node, this selector is formed as follows.
//!
//! ```text
//! i = popcnt(v & ((2 << n) - 1))
//! ```
//!
//! This turns all the zeros to the right of the selector (or put differently,
//! in positions of lesser significance) into ones. The population count
//! (`popcnt`) instruction then counts those ones. This gives us a partial index
//! into the interior nodes array. Because the index has a maximum value of 64,
//! this can only be a relative index as there will be many more than 64
//! interior and leaf nodes in a real tree. To overcome this, interior nodes
//! contain an offset to combine with the index to find the correct position in
//! the corresponding array.
//!
//! ```
//! pub struct Interior {
//!     pub iv: u64,
//!     pub interior_offset: u64,
//!     pub leaf_offset: u64,
//! }
//! ```
//!
//! The complete index is then
//!
//! ```text
//! i += interior_offset
//! ```
//!
//! In the leaf-node case where the bitwise intersection of `sel` and `v` is
//! zero we need to form the index a bit differently as the above `popcnt` will
//! just equal zero. This is easily done by flipping the bits in `v` and then
//! applying the same logic. This has the effect of counting the zeros to the
//! right of the one in `sel`.
//!
//! ```text
//! i = popcnt(!v & ((2 << n) - 1))
//! ```
//!
//! **TODO(ry):** is the bitwise and really necessary in the `popcnt`
//! calculation.  We already know from the selector test whether or not there is
//! an intersection so should we be able to just do `popcnt((2 << n) - 1)`? in
//! either the interior or leaf case?
//!
//! To complete our root poptrie node above, we need to add offsets. In this
//! case, since it's the root node and there are no other nodes yet we can set
//! the `interior_offset` to `1` (accounting for the root node itself) and the
//! `leaf_offset` to `0`.
//!
//! ```text
//!     6    4    1    1
//!     1    2    9    2    0  
//! +-------------------------+
//! |..|1|..|1|..|1|..|1|..|1|| interior_offset=1 leaf_offset=0
//! +-------------------------+
//! ```
//!
//! We need a leaf node for the default route. In this example we'll have leaf
//! nodes contain nexthops directly. Therefore the single leaf node contained by
//! the root node is the following (the default route). This route is not
//! pointed to by any particular child index, as the prefix length is zero, so
//! this is a bit of a special case.
//!
//! ```text
//!         +---------+
//! null -> | 1.2.3.4 |
//!         +---------+
//! ```
//!
//! In order to determine the rest of the nodes, it will be useful to break out
//! the prefixes in the routing table into 6-bit segments.
//!
//! ```text
//! Destination     Segments                              6-bit values
//! -----------     --------                              ------------
//! 1.0.0.0/8       000000_010000_000000_000000_000000_00 |0   16              |
//! 247.33.0.0/16   111101_110010_000100_000000_000000_00 |61  50  4           |
//! 247.33.12.0/24  111101_110010_000100_001100_000000_00 |61  50  4   12      |
//! 51.12.109.0/24  001100_110000_110001_101101_000000_00 |12  48  49  45      |
//! 77.18.0.0/16    010011_010001_001000_000000_000000_00 |19  17  8           |
//! 170.1.14.3/32   101010_100000_000100_001110_000000_11 |42  32  4   14  0  3|
//! 0.0.0.0.0/0     000000_000000_000000_000000_000000_00 |                    |
//! ```
//!
//! From this table we can see that the first column corresponds directly to the
//! bitvec structure of the root node. We've already covered the single leaf
//! child node for the root. What remains are 5 child internal nodes stemming
//! from bitvec positions 0, 61, 12, 19 and 42. These yield the following
//! internal child nodes.
//!
//! ```text
//!         1              5              4              1              3
//!         6              0              8              7              2
//!     +-------+      +-------+      +-------+      +-------+      +-------+
//!  0: |..|1|..|  61: |..|1|..|  12: |..|1|..|  19: |..|1|..|  42: |..|1|..|
//!     +-------+      +-------+      +-------+      +-------+      +-------+
//! ```
//!
//! Taking these internal child nodes in turn. The following is for the first
//! routing table entry.
//!  
//! ```text
//!          1
//!          6
//!      +-------+        +---------------+
//!   0: |..|0|..|   ,--> | 1.254.254.254 |
//!      +-------+   |    +---------------+
//!          `-------'
//! ```
//!
//! At this point we have consumed 12 bits of the input IP address. This is
//! beyond the 8 bit prefix so we need a leaf entry for that prefix. That leaf
//! node is shown to the right of the internal node in the depiction above.
//!
//! The next internal node represents a pair of prefixes in the routing table,
//! `247.33.0.0/16` and `247.33.12.0/24`. We do not have enough bits for a leaf
//! node on either, so we have another internal node
//!
//! ```text
//!          5
//!          0               4
//!      +-------+       +-------+
//!  61: |..|1|..|   ,-->|..|1|..|
//!      +-------+   |   +-------+
//!          `-------'
//!```
//! At the next internal node
//!
//!```text
//!          1
//!          2           +-------------+
//!          ,---------->| 247.33.12.1 |
//!      +-------+       +-------------+
//!   4: |..|0|..|
//!      +-------+       +------------+
//!        `----`------->| 247.33.0.1 |
//!                      +------------+
//! ```
//!
//! For the entry `51.12.109.0/24` we have:
//!
//! ```text
//!          4               4
//!          8               9
//!      +-------+       +-------+
//!  12: |..|1|..|   ,-->|..|1|..|
//!      +-------+   |   +-------+
//!          `-------'
//! ```
//!
//! For the entry `77.18.0.0/16` we have:
//!
//! ```text
//!          1
//!          7               8
//!      +-------+       +-------+
//!  19: |..|1|..|   ,-->|..|1|..|
//!      +-------+   |   +-------+
//!          `-------'
//! ```
//!
//! For the entry `170.1.14.3/32` we have:
//!
//! ```text
//!          3
//!          2               4
//!      +-------+       +-------+
//!  42: |..|1|..|   ,-->|..|1|..|
//!      +-------+   |   +-------+
//!          `-------'
//! ```
//!

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::{BitAnd, Shl, Shr};
use util::mask_through;

mod util;

#[cfg(test)]
mod test;

#[cfg(test)]
#[macro_use]
extern crate std;

/// The poptrie data structure.
#[derive(Debug)]
pub struct Poptrie<T> {
    /// An array of interior nodes.
    pub interior: Vec<Interior>,

    /// An array of leaf nodes.
    pub leaf: Vec<Leaf<T>>,

    /// A default route if any.
    pub default: Option<Leaf<T>>,
}

/// An interior poptrie node.
pub struct Interior {
    /// The bit vector that describes child internal nodes.
    pub iv: u64,

    /// The bit vector that describes child leaf nodes.
    pub lv: u64,

    /// An offset into Poptrie::interior where the child interior nodes of this
    /// node begin.
    pub interior_offset: u64,

    /// An offset into Poptrie::interior where the child leaf nodes of this node
    /// begin.
    pub leaf_offset: u64,
}

/// A leaf poptrie node.
#[derive(Debug)]
pub struct Leaf<T> {
    /// The data associated with this node.
    pub data: T,
}

pub trait IpAddress:
    Sized
    + BitAnd<Output = Self>
    + Shr<u8, Output = Self>
    + Shl<u8, Output = Self>
    + From<u8>
    + Copy
{
    const BITS: u8;
    const BYTES: usize = (Self::BITS / 8) as usize;
    type ByteArray: AsRef<[u8]> + AsMut<[u8]> + Copy + Ord;

    fn from_be_bytes(bytes: &Self::ByteArray) -> Self;
    fn to_be_bytes(self) -> Self::ByteArray;
    fn to_u8(self) -> u8;
}

impl IpAddress for u32 {
    const BITS: u8 = u32::BITS as u8;
    type ByteArray = [u8; Self::BYTES];

    fn from_be_bytes(bytes: &Self::ByteArray) -> Self {
        u32::from_be_bytes(*bytes)
    }

    fn to_be_bytes(self) -> Self::ByteArray {
        self.to_be_bytes()
    }

    #[inline]
    fn to_u8(self) -> u8 {
        self as u8
    }
}

impl IpAddress for u128 {
    const BITS: u8 = u128::BITS as u8;
    type ByteArray = [u8; Self::BYTES];

    fn from_be_bytes(bytes: &Self::ByteArray) -> Self {
        u128::from_be_bytes(*bytes)
    }

    fn to_be_bytes(self) -> Self::ByteArray {
        self.to_be_bytes()
    }

    #[inline]
    fn to_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Debug)]
pub struct IpRoutingTable<Ip: IpAddress, T>(
    pub BTreeMap<(Ip::ByteArray, u8), T>,
);

impl<Ip: IpAddress, T> IpRoutingTable<Ip, T> {
    pub fn add(&mut self, dst: Ip::ByteArray, len: u8, nexthop: T) {
        self.0.insert((dst, len), nexthop);
    }
}

#[derive(Clone, Debug)]
pub struct Ipv4RoutingTable<T>(pub BTreeMap<([u8; 4], u8), T>);

impl<T> Ipv4RoutingTable<T> {
    pub fn add(&mut self, dst: [u8; 4], len: u8, nexthop: T) {
        self.0.insert((dst, len), nexthop);
    }
}

impl<T: Clone> From<Ipv4RoutingTable<T>> for Poptrie<T> {
    fn from(tree: Ipv4RoutingTable<T>) -> Self {
        let mut s = Self::default();
        s.construct4(tree);
        s
    }
}

#[derive(Clone, Debug)]
pub struct Ipv6RoutingTable<T>(pub BTreeMap<([u8; 16], u8), T>);

impl<T> Ipv6RoutingTable<T> {
    pub fn add(&mut self, dst: [u8; 16], len: u8, nexthop: T) {
        self.0.insert((dst, len), nexthop);
    }
}

impl<T: Clone> From<Ipv6RoutingTable<T>> for Poptrie<T> {
    fn from(tree: Ipv6RoutingTable<T>) -> Self {
        let mut s = Self::default();
        s.construct6(tree);
        s
    }
}

fn matcher<Ip: IpAddress, T: Clone>(
    poptrie: &Poptrie<T>,
    addr: Ip,
) -> Option<T> {
    let mut i = 0u64;
    let mut v = poptrie.interior.get(i as usize)?.iv;
    let mut offset = 0;
    let mut n = crate::util::extract(6, offset, addr);

    let mut result = None;

    #[cfg(test)]
    println!("n={n}");

    #[cfg(test)]
    println!("{:#?}", poptrie.interior.get(i as usize)?);

    while (v & (1u64 << n)) != 0 {
        // Check for stash at CURRENT node BEFORE descending.
        // This handles the case where both an interior child AND a leaf
        // exist at the same position (shorter prefix provides fallback).
        {
            let lv = poptrie.interior.get(i as usize)?.lv;
            if (lv & (1u64 << n)) != 0 {
                let base = poptrie.interior.get(i as usize)?.leaf_offset;
                let arg = lv & mask_through(n);
                let bc = arg.count_ones() as u64;
                let leaf_i = base - lv.count_ones() as u64 + bc - 1;
                result = Some(poptrie.leaf.get(leaf_i as usize)?.data.clone())
            }
        }

        // Now descend to the interior child
        let base = poptrie.interior.get(i as usize)?.interior_offset;
        let arg = v & mask_through(n);
        let bc = arg.count_ones() as u64;
        i = base + bc - 1;
        v = poptrie.interior.get(i as usize)?.iv;

        offset += 1;
        n = crate::util::extract(6, offset, addr);

        #[cfg(test)]
        println!("n={n}");

        #[cfg(test)]
        println!("{:#?}", poptrie.interior.get(i as usize)?);
    }

    // Final leaf lookup at the terminal node
    let base = poptrie.interior.get(i as usize)?.leaf_offset;
    let v = poptrie.interior.get(i as usize)?.lv;
    if (v & (1u64 << n)) != 0 {
        let arg = v & mask_through(n);
        let bc = arg.count_ones() as u64;
        i = base - v.count_ones() as u64 + bc - 1;
        result = Some(poptrie.leaf.get(i as usize)?.data.clone())
    }

    result
}

fn construct<Ip: IpAddress, T: Clone>(
    poptrie: &mut Poptrie<T>,
    tree: IpRoutingTable<Ip, T>,
) {
    let depth = Ip::BITS.div_ceil(6);
    let bits = Ip::BITS;
    let mut forest = vec![(0, tree)];

    let mut ioff = 1;
    for depth in 0..depth {
        let mut subforest = Vec::<(u8, IpRoutingTable<Ip, T>)>::new();
        // Tracks cumulative count of children from preceding sibling trees.
        // This is used to compute interior_offset for each tree's children.
        let mut child_offset = 0;
        for (_, tree) in &forest {
            let mut iv = 0u64;
            let mut lv = 0u64;

            let mut subsubforest = Vec::<(u8, IpRoutingTable<Ip, T>)>::new();
            // Collect leaves with their bit positions for later sorting.
            // Leaves must be pushed in bit-position order for the matcher's
            // popcount-based indexing to work correctly.
            let mut pending_leaves: Vec<(u8, T)> = Vec::new();
            // Sort routes by descending prefix length (more specific first).
            // This ensures longer prefixes claim their slots before shorter
            // prefixes, implementing proper longest-prefix-match semantics.
            let mut routes: Vec<_> = tree.0.iter().collect();
            routes.sort_by(|a, b| b.0 .1.cmp(&a.0 .1));

            for ((addr, prefix_len), data) in routes {
                // default route case
                if *prefix_len == 0 {
                    poptrie.default = Some(Leaf { data: data.clone() });
                    continue;
                }
                let k = crate::util::extract(6, depth, Ip::from_be_bytes(addr));
                let consumed = core::cmp::min((depth + 1) * 6, bits);
                if *prefix_len <= consumed {
                    // Only add leaf if no existing leaf at this position.
                    // More specific prefixes (processed first) take precedence.
                    // Note: We allow setting lv even when iv is set - this provides
                    // fallback matches when traversing through interior nodes
                    // doesn't find a more specific match (the "stash" logic).
                    if ((1u64 << k) & lv) == 0 {
                        lv |= 1u64 << k;
                        pending_leaves.push((k, data.clone()));
                    }

                    // If the prefix of the router entry is less than but not equal
                    // to the consumed number of bits, we need to add those bits to
                    // the bitvec.
                    if *prefix_len != consumed {
                        // Shift by the extra bits and add to the bitvec for this
                        // internal node.
                        let extra = 1u64 << (consumed - *prefix_len);
                        for i in 1..(extra) {
                            // Only add if slot not already claimed by more specific prefix
                            if ((1u64 << (k + i as u8)) & lv) == 0 {
                                lv |= 1u64 << (k + i as u8);
                                pending_leaves
                                    .push((k + i as u8, data.clone()));
                            }
                        }
                    }
                    continue;
                }
                iv |= 1u64 << k;
                match subsubforest.iter_mut().find(|x| x.0 == k) {
                    Some(ref mut entry) => {
                        entry.1.add(*addr, *prefix_len, data.clone());
                    }
                    None => {
                        let mut tbl = IpRoutingTable::default();
                        tbl.add(*addr, *prefix_len, data.clone());
                        subsubforest.push((k, tbl));
                    }
                }
            }

            // Sort leaves by bit position and push them in order.
            // The matcher uses popcount to index into leaves, which requires
            // leaves to be stored in bit-position order.
            pending_leaves.sort_by(|a, b| a.0.cmp(&b.0));
            for (_, data) in pending_leaves {
                poptrie.leaf.push(Leaf { data });
            }

            if iv > 0 || lv > 0 {
                poptrie.interior.push(Interior {
                    iv,
                    lv,
                    interior_offset: if iv > 0 {
                        ioff + child_offset
                    } else {
                        0
                    },
                    leaf_offset: poptrie.leaf.len() as u64,
                });
            }
            // Sort children by k value to maintain correct popcount-based indexing.
            // The matcher uses popcount to find the child index, which assumes
            // children are stored in order of their bit positions in iv.
            subsubforest.sort_by(|a, b| a.0.cmp(&b.0));

            // Add this tree's children count to the offset for subsequent trees
            child_offset += subsubforest.len() as u64;
            subforest.extend_from_slice(&subsubforest);
        }
        ioff += subforest.len() as u64;
        forest = subforest;
    }
}

impl<T: Clone> Poptrie<T> {
    pub fn construct4(&mut self, tree: Ipv4RoutingTable<T>) {
        construct(self, IpRoutingTable::<u32, T>(tree.0));
    }

    pub fn construct6(&mut self, tree: Ipv6RoutingTable<T>) {
        construct(self, IpRoutingTable::<u128, T>(tree.0));
    }

    pub fn match_v4(&self, addr: u32) -> Option<T> {
        self.do_match_v4(addr)
            .or(self.default.as_ref().map(|x| x.data.clone()))
    }

    pub fn match_v6(&self, addr: u128) -> Option<T> {
        self.do_match_v6(addr)
            .or(self.default.as_ref().map(|x| x.data.clone()))
    }

    pub fn do_match_v4(&self, addr: u32) -> Option<T> {
        matcher(self, addr)
    }

    pub fn do_match_v6(&self, addr: u128) -> Option<T> {
        matcher(self, addr)
    }
}
