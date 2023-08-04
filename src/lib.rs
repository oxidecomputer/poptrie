// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Copyright 2023 Oxide Computer Company

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

// NOTE #[derive(Default)] see:
//     broken https://github.com/rust-lang/rust/issues/26925
impl<T> Default for Poptrie<T> {
    fn default() -> Self {
        Self {
            interior: Vec::new(),
            leaf: Vec::new(),
            default: None,
        }
    }
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

impl core::fmt::Debug for Interior {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut islots = Vec::new();
        let mut lslots = Vec::new();
        for i in 0..63 {
            if (self.iv & (1 << i)) != 0 {
                islots.push(i);
            }
        }
        for i in 0..63 {
            if (self.lv & (1 << i)) != 0 {
                lslots.push(i);
            }
        }
        f.debug_struct("Interior")
            .field("iv", &islots)
            .field("lv", &lslots)
            .field("interior_offset", &self.interior_offset)
            .field("leaf_offset", &self.leaf_offset)
            .finish()
    }
}

/// A leaf poptrie node.
#[derive(Debug)]
pub struct Leaf<T> {
    /// The data associated with this node.
    pub data: T,
}

#[derive(Clone)]
pub struct Ipv4RoutingTable<T>(pub BTreeMap<(u32, u8), T>);

// NOTE #[derive(Default)] see:
//     broken https://github.com/rust-lang/rust/issues/26925
impl<T> Default for Ipv4RoutingTable<T> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<T> core::ops::Deref for Ipv4RoutingTable<T> {
    type Target = BTreeMap<(u32, u8), T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> core::ops::DerefMut for Ipv4RoutingTable<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Ipv4RoutingTable<T> {
    pub fn add(&mut self, dst: [u8; 4], len: u8, nexthop: T) {
        self.0.insert((u32::from_be_bytes(dst), len), nexthop);
    }
}

impl<T: Copy> From<Ipv4RoutingTable<T>> for Poptrie<T> {
    fn from(tree: Ipv4RoutingTable<T>) -> Self {
        let mut s = Self::default();
        s.construct4(tree);
        s
    }
}

#[derive(Clone)]
pub struct Ipv6RoutingTable<T>(pub BTreeMap<(u128, u8), T>);

// NOTE #[derive(Default)] see:
//     broken https://github.com/rust-lang/rust/issues/26925
impl<T> Default for Ipv6RoutingTable<T> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<T> core::ops::Deref for Ipv6RoutingTable<T> {
    type Target = BTreeMap<(u128, u8), T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> core::ops::DerefMut for Ipv6RoutingTable<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Ipv6RoutingTable<T> {
    pub fn add(&mut self, dst: [u8; 16], len: u8, nexthop: T) {
        self.0.insert((u128::from_be_bytes(dst), len), nexthop);
    }
}

impl<T: Copy> From<Ipv6RoutingTable<T>> for Poptrie<T> {
    fn from(tree: Ipv6RoutingTable<T>) -> Self {
        let mut s = Self::default();
        s.construct6(tree);
        s
    }
}

macro_rules! extract {
    ($width:expr, $offset:expr, $v:expr, $bits:expr) => {{
        let shift = $bits.saturating_sub($width * ($offset + 1));
        let mask = 0b111111 << shift;
        let res = ($v & mask) >> shift;
        res as u8
    }};
}

pub fn extract_32(width: u8, offset: u8, v: u32) -> u8 {
    extract!(width, offset, v, 32u8)
}

pub fn extract_128(width: u8, offset: u8, v: u32) -> u8 {
    extract!(width, offset, v, 128u8)
}

//TODO having this as a macro is terrible for debugging as we get no backtrace
macro_rules! matcher {
    ($self:ident, $addr:tt, $bits:expr) => {{
        let mut i = 0u64;
        let mut v = $self.interior[i as usize].iv;
        let mut offset = 0;
        let mut n = extract!(6, offset, $addr, $bits);

        #[cfg(test)]
        println!("n={n}");

        #[cfg(test)]
        println!("{:#?}", $self.interior[i as usize]);

        let mut result = $self.default.as_ref().map(|x| x.data);

        while (v & (1 << n)) != 0 {
            let base = $self.interior[i as usize].interior_offset;
            let arg = v & ((2 << n) - 1);
            let bc = arg.count_ones() as u64;
            i = base + bc - 1;
            v = $self.interior[i as usize].iv;

            offset += 1;
            n = extract!(6, offset, $addr, $bits);

            #[cfg(test)]
            println!("n={n}");

            #[cfg(test)]
            println!("{:#?}", $self.interior[i as usize]);

            // check for stash any potentially suboptimal matches, longer
            // prefix matches will overwrite these
            let base = $self.interior[i as usize].leaf_offset;
            let v = $self.interior[i as usize].lv;
            if (v & (1 << n)) != 0 {
                let i = base - 1;
                result = Some($self.leaf[i as usize].data)
            }
        }

        let base = $self.interior[i as usize].leaf_offset;
        let v = $self.interior[i as usize].lv;
        if (v & (1 << n)) != 0 {
            i = base - 1;
            result = Some($self.leaf[i as usize].data)
        }

        result
    }};
}

//TODO having this as a macro is terrible for debugging as we get no backtrace
macro_rules! construct {
    ($self:ident, $tree:ident, $bits:expr, $depth:expr, $rt:ident<$t:tt>) => {{
        let mut forest = vec![(0, $tree)];

        let mut ioff = 1;
        for depth in 0..$depth {
            let mut subforest = Vec::<(u8, $rt<$t>)>::new();
            let mut children = 0;
            let mut siblings = 0;
            for (_, $tree) in &forest {
                let mut iv = 0u64;
                let mut lv = 0u64;

                let mut subsubforest = Vec::<(u8, $rt<$t>)>::new();
                for (r, e) in &$tree.0 {
                    // default route case
                    if r.1 == 0 {
                        $self.default = Some(Leaf { data: *e });
                        continue;
                    }
                    let k = extract!(6, depth, r.0, $bits);
                    let consumed = core::cmp::min((depth + 1) * 6, $bits);
                    if r.1 <= consumed {
                        if ((1 << k) & iv) == 0 {
                            lv |= 1 << k;
                            $self.leaf.push(Leaf { data: *e });
                        }

                        // If the prefix of the router entry is less than but not equal
                        // to the consumed number of bits, we need to add those bits to
                        // the bitvec.
                        if r.1 != consumed {
                            // Shift by the extra bits and add to the bitvec for this
                            // internal node.
                            let extra = 1 << (consumed - r.1);
                            for i in 1..(extra) {
                                lv |= 1 << (k + i);
                                $self.leaf.push(Leaf { data: *e });
                            }
                        }
                        continue;
                    }
                    iv |= 1 << k;
                    match subsubforest.iter_mut().find(|x| x.0 == k) {
                        Some(ref mut entry) => {
                            entry.1.insert(*r, *e);
                        }
                        None => {
                            let mut tbl = $rt::<$t>::default();
                            tbl.insert(*r, *e);
                            subsubforest.push((k, tbl));
                            if iv > 0 {
                                children += 1;
                            }
                        }
                    }
                }

                if iv > 0 || lv > 0 {
                    $self.interior.push(Interior {
                        iv,
                        lv,
                        interior_offset: if iv > 0 {
                            ioff + siblings
                        } else {
                            0
                        },
                        leaf_offset: $self.leaf.len() as u64,
                    });
                    if iv > 0 {
                        siblings += 1;
                    }
                }
                subforest.extend_from_slice(&subsubforest);
            }
            ioff += children;
            forest = subforest;
        }
    }}
}

impl<T: Copy> Poptrie<T> {
    pub fn construct4(&mut self, tree: Ipv4RoutingTable<T>) {
        construct!(self, tree, 32u8, 6, Ipv4RoutingTable<T>);
    }

    pub fn construct6(&mut self, tree: Ipv6RoutingTable<T>) {
        construct!(self, tree, 128u8, 22, Ipv6RoutingTable<T>);
    }

    pub fn match_v4(&self, addr: u32) -> Option<T> {
        matcher!(self, addr, 32u8)
    }

    pub fn match_v6(&self, addr: u128) -> Option<T> {
        matcher!(self, addr, 128u8)
    }
}

#[cfg(test)]
#[macro_use]
extern crate std;

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Default, Copy, Clone, PartialEq)]
    struct Ipv4(u32);
    impl core::fmt::Debug for Ipv4 {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let b = self.0.to_be_bytes();
            write!(f, "{}.{}.{}.{}", b[0], b[1], b[2], b[3])
        }
    }

    impl Ipv4 {
        fn new(v: [u8; 4]) -> Self {
            Self(u32::from_be_bytes(v))
        }
    }

    #[derive(Default, Copy, Clone, PartialEq)]
    struct Ipv6(u128);
    impl core::fmt::Debug for Ipv6 {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let b = self.0.to_be_bytes();
            write!(
                f,
                "{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}",
                b[0], b[1], b[2], b[3],
                b[4], b[5], b[6], b[7],
                b[8], b[9], b[10], b[11],
                b[12], b[13], b[14], b[15],
            )
        }
    }

    impl Ipv6 {
        fn new(v: [u8; 16]) -> Self {
            Self(u128::from_be_bytes(v))
        }
    }

    impl std::str::FromStr for Ipv6 {
        type Err = std::net::AddrParseError;
        fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
            let addr: std::net::Ipv6Addr = s.parse()?;
            Ok(Self::new(addr.octets()))
        }
    }

    #[test]
    fn test_extract32() {
        // Verify documentation examples

        // 1.0.0.0
        let v = u32::from_be_bytes([1, 0, 0, 0]);
        let x = extract_32_all(v);
        assert_eq!(x, [0, 16, 0, 0, 0, 0]);

        // 247.33.0.0
        let v = u32::from_be_bytes([247, 33, 0, 0]);
        let x = extract_32_all(v);
        assert_eq!(x, [61, 50, 4, 0, 0, 0]);

        // 247.33.12.0
        let v = u32::from_be_bytes([247, 33, 12, 0]);
        let x = extract_32_all(v);
        assert_eq!(x, [61, 50, 4, 12, 0, 0]);

        // 51.12.109.0
        let v = u32::from_be_bytes([51, 12, 109, 0]);
        let x = extract_32_all(v);
        assert_eq!(x, [12, 48, 49, 45, 0, 0]);

        // 77.18.0.0
        let v = u32::from_be_bytes([77, 18, 0, 0]);
        let x = extract_32_all(v);
        assert_eq!(x, [19, 17, 8, 0, 0, 0]);

        // 170.1.14.3
        let v = u32::from_be_bytes([170, 1, 14, 3]);
        let x = extract_32_all(v);
        assert_eq!(x, [42, 32, 4, 14, 0, 3]);

        // 0.0.0.0
        let v = u32::from_be_bytes([0, 0, 0, 0]);
        let x = extract_32_all(v);
        assert_eq!(x, [0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_construct_rec() {
        let tbl = test_routing_table_with_default_route_v4();
        let pt = Poptrie::<Ipv4>::from(tbl);

        #[allow(clippy::identity_op)]
        let expected_root_bitvec =
            0u64 | 1 << 0 | 1 << 61 | 1 << 61 | 1 << 12 | 1 << 19 | 1 << 42;

        assert_eq!(expected_root_bitvec, pt.interior[0].iv);
        assert_eq!(pt.leaf.len(), 27);

        println!("{:#?}", pt);
    }

    #[test]
    fn test_match_v4() {
        let tbl = test_routing_table_v4();
        let pt = Poptrie::<Ipv4>::from(tbl);

        // Test hits
        let addr = Ipv4::new([1, 7, 0, 1]);
        let m = pt.match_v4(addr.0);
        assert_eq!(m, Some(Ipv4::new([1, 254, 254, 254])));

        let addr = Ipv4::new([247, 33, 0, 1]);
        let m = pt.match_v4(addr.0);
        assert_eq!(m, Some(Ipv4::new([247, 33, 0, 1])));

        let addr = Ipv4::new([247, 33, 12, 1]);
        let m = pt.match_v4(addr.0);
        assert_eq!(m, Some(Ipv4::new([247, 33, 12, 1])));

        let addr = Ipv4::new([51, 12, 109, 1]);
        let m = pt.match_v4(addr.0);
        assert_eq!(m, Some(Ipv4::new([51, 12, 109, 10])));

        let addr = Ipv4::new([77, 18, 4, 7]);
        let m = pt.match_v4(addr.0);
        assert_eq!(m, Some(Ipv4::new([77, 18, 10, 1])));

        let addr = Ipv4::new([170, 1, 14, 3]);
        let m = pt.match_v4(addr.0);
        assert_eq!(m, Some(Ipv4::new([1, 7, 0, 1])));

        // Test default route
        let addr = Ipv4::new([4, 7, 0, 1]);
        let m = pt.match_v4(addr.0);
        assert_eq!(m, None);

        let tbl = test_routing_table_with_default_route_v4();
        let pt = Poptrie::<Ipv4>::from(tbl);

        // Test default route
        let addr = Ipv4::new([4, 7, 0, 1]);
        let m = pt.match_v4(addr.0);
        assert_eq!(m, Some(Ipv4::new([1, 2, 3, 4])));
    }

    #[test]
    fn test_match_v6() {
        let tbl = test_routing_table_v6();
        let pt = Poptrie::<Ipv6>::from(tbl);

        // Test hits
        let addr: Ipv6 = "1:7:0::1".parse().unwrap();
        let m = pt.match_v6(addr.0);
        let gw: Ipv6 = "1::ffff:ffff:ffff".parse().unwrap();
        assert_eq!(m, Some(gw));

        let addr: Ipv6 = "247:33::1".parse().unwrap();
        let m = pt.match_v6(addr.0);
        let gw: Ipv6 = "247:33::1".parse().unwrap();
        assert_eq!(m, Some(gw));

        let addr: Ipv6 = "247:33:12::1".parse().unwrap();
        let m = pt.match_v6(addr.0);
        let gw: Ipv6 = "247:33:12::1".parse().unwrap();
        assert_eq!(m, Some(gw));

        let addr: Ipv6 = "51:12:109::1".parse().unwrap();
        let m = pt.match_v6(addr.0);
        let gw: Ipv6 = "51:12:109::10".parse().unwrap();
        assert_eq!(m, Some(gw));

        let addr: Ipv6 = "77:18:4::7".parse().unwrap();
        let m = pt.match_v6(addr.0);
        let gw: Ipv6 = "77:18:10::1".parse().unwrap();
        assert_eq!(m, Some(gw));

        let addr: Ipv6 = "170:1:14::3".parse().unwrap();
        let m = pt.match_v6(addr.0);
        let gw: Ipv6 = "1:7:0::1".parse().unwrap();
        assert_eq!(m, Some(gw));

        // Test default route
        let addr: Ipv6 = "4:7:0::1".parse().unwrap();
        let m = pt.match_v6(addr.0);
        assert_eq!(m, None);

        let tbl = test_routing_table_with_default_route_v6();
        let pt = Poptrie::<Ipv6>::from(tbl);

        let addr: Ipv6 = "4:7:0::1".parse().unwrap();
        let m = pt.match_v6(addr.0);
        let gw: Ipv6 = "1:2:3::4".parse().unwrap();
        assert_eq!(m, Some(gw));
    }

    fn test_routing_table_v4() -> Ipv4RoutingTable<Ipv4> {
        let mut tbl = Ipv4RoutingTable::<Ipv4>::default();
        tbl.add([1, 0, 0, 0], 8, Ipv4::new([1, 254, 254, 254]));
        tbl.add([247, 33, 0, 0], 16, Ipv4::new([247, 33, 0, 1]));
        tbl.add([247, 33, 12, 0], 24, Ipv4::new([247, 33, 12, 1]));
        tbl.add([51, 12, 109, 0], 24, Ipv4::new([51, 12, 109, 10]));
        tbl.add([77, 18, 0, 0], 16, Ipv4::new([77, 18, 10, 1]));
        tbl.add([170, 1, 14, 3], 32, Ipv4::new([1, 7, 0, 1]));
        tbl
    }

    fn test_routing_table_with_default_route_v4() -> Ipv4RoutingTable<Ipv4> {
        let mut tbl = test_routing_table_v4();
        tbl.add([0, 0, 0, 0], 0, Ipv4::new([1, 2, 3, 4]));
        tbl
    }

    fn test_routing_table_v6() -> Ipv6RoutingTable<Ipv6> {
        let mut tbl = Ipv6RoutingTable::<Ipv6>::default();

        let rt: std::net::Ipv6Addr = "1::".parse().unwrap();
        let gw: Ipv6 = "1::ffff:ffff:ffff".parse().unwrap();
        tbl.add(rt.octets(), 16, gw);

        let rt: std::net::Ipv6Addr = "247:33::".parse().unwrap();
        let gw: Ipv6 = "247:33::1".parse().unwrap();
        tbl.add(rt.octets(), 32, gw);

        let rt: std::net::Ipv6Addr = "247:33:12::".parse().unwrap();
        let gw: Ipv6 = "247:33:12::1".parse().unwrap();
        tbl.add(rt.octets(), 48, gw);

        let rt: std::net::Ipv6Addr = "51:12:109::".parse().unwrap();
        let gw: Ipv6 = "51:12:109::10".parse().unwrap();
        tbl.add(rt.octets(), 48, gw);

        let rt: std::net::Ipv6Addr = "77:18::".parse().unwrap();
        let gw: Ipv6 = "77:18:10::1".parse().unwrap();
        tbl.add(rt.octets(), 32, gw);

        let rt: std::net::Ipv6Addr = "170:1:14::3".parse().unwrap();
        let gw: Ipv6 = "1:7:0::1".parse().unwrap();
        tbl.add(rt.octets(), 128, gw);

        tbl
    }

    fn test_routing_table_with_default_route_v6() -> Ipv6RoutingTable<Ipv6> {
        let mut tbl = test_routing_table_v6();

        let rt: std::net::Ipv6Addr = "::".parse().unwrap();
        let gw: Ipv6 = "1:2:3::4".parse().unwrap();
        tbl.add(rt.octets(), 0, gw);

        tbl
    }

    fn extract_32_all(v: u32) -> [u8; 6] {
        [
            extract_32(6, 0, v),
            extract_32(6, 1, v),
            extract_32(6, 2, v),
            extract_32(6, 3, v),
            extract_32(6, 4, v),
            extract_32(6, 5, v),
        ]
    }
}
