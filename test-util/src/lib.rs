//! Test utilities for poptrie.
//!
//! This crate provides common testing utilities including:
//! - Prefix masking functions for IPv4 and IPv6
//! - Naive longest-prefix-match implementations for verification
//! - Table and address generation utilities for benchmarks
//! - Proptest strategies (behind the `proptest` feature)

use poptrie::{Ipv4RoutingTable, Ipv6RoutingTable};

/// Apply a prefix mask to an IPv4 address.
///
/// Returns the address with host bits (beyond prefix_len) zeroed out.
pub fn mask_v4(addr: [u8; 4], prefix_len: u8) -> [u8; 4] {
    let addr_u32 = u32::from_be_bytes(addr);
    let mask = if prefix_len == 0 {
        0
    } else {
        !0u32 << (32 - prefix_len)
    };
    (addr_u32 & mask).to_be_bytes()
}

/// Apply a prefix mask to an IPv6 address.
///
/// Returns the address with host bits (beyond prefix_len) zeroed out.
pub fn mask_v6(addr: [u8; 16], prefix_len: u8) -> [u8; 16] {
    let addr_u128 = u128::from_be_bytes(addr);
    let mask = if prefix_len == 0 {
        0
    } else {
        !0u128 << (128 - prefix_len)
    };
    (addr_u128 & mask).to_be_bytes()
}

/// Check if addr is within the prefix defined by (prefix_addr, prefix_len).
pub fn addr_in_prefix_v4(addr: [u8; 4], prefix_addr: [u8; 4], prefix_len: u8) -> bool {
    mask_v4(addr, prefix_len) == mask_v4(prefix_addr, prefix_len)
}

/// Check if addr is within the prefix defined by (prefix_addr, prefix_len).
pub fn addr_in_prefix_v6(addr: [u8; 16], prefix_addr: [u8; 16], prefix_len: u8) -> bool {
    mask_v6(addr, prefix_len) == mask_v6(prefix_addr, prefix_len)
}

/// Find the longest matching prefix for an address in a routing table.
///
/// This is a naive O(n) implementation used for verification against
/// the optimized poptrie implementation.
pub fn longest_match_v4<T: Clone>(table: &Ipv4RoutingTable<T>, addr: [u8; 4]) -> Option<T> {
    let mut best_match: Option<(u8, T)> = None;
    for ((prefix_addr, prefix_len), nexthop) in table.iter() {
        if addr_in_prefix_v4(addr, *prefix_addr, *prefix_len) {
            match &best_match {
                None => best_match = Some((*prefix_len, nexthop.clone())),
                Some((best_len, _)) if prefix_len > best_len => {
                    best_match = Some((*prefix_len, nexthop.clone()))
                }
                _ => {}
            }
        }
    }
    best_match.map(|(_, nh)| nh)
}

/// Find the longest matching prefix for an address in a routing table.
///
/// This is a naive O(n) implementation used for verification against
/// the optimized poptrie implementation.
pub fn longest_match_v6<T: Clone>(table: &Ipv6RoutingTable<T>, addr: [u8; 16]) -> Option<T> {
    let mut best_match: Option<(u8, T)> = None;
    for ((prefix_addr, prefix_len), nexthop) in table.iter() {
        if addr_in_prefix_v6(addr, *prefix_addr, *prefix_len) {
            match &best_match {
                None => best_match = Some((*prefix_len, nexthop.clone())),
                Some((best_len, _)) if prefix_len > best_len => {
                    best_match = Some((*prefix_len, nexthop.clone()))
                }
                _ => {}
            }
        }
    }
    best_match.map(|(_, nh)| nh)
}

/// Generate a deterministic routing table with n routes.
///
/// Produces varied prefixes across the address space with prefix lengths
/// ranging from /8 to /32.
pub fn generate_table(n: usize) -> Ipv4RoutingTable<u32> {
    let mut table = Ipv4RoutingTable::default();
    for i in 0..n {
        let a = ((i * 7) % 256) as u8;
        let b = ((i * 13) % 256) as u8;
        let c = ((i * 17) % 256) as u8;
        let d = ((i * 23) % 256) as u8;
        let prefix_len = (8 + (i % 25)) as u8; // /8 to /32
        let addr = mask_v4([a, b, c, d], prefix_len);
        table.add(addr, prefix_len, i as u32);
    }
    table
}

/// Generate deterministic lookup addresses.
///
/// Produces n addresses spread across the address space in a deterministic
/// but pseudo-random pattern.
pub fn generate_addrs(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let a = ((i * 31) % 256) as u8;
            let b = ((i * 37) % 256) as u8;
            let c = ((i * 41) % 256) as u8;
            let d = ((i * 43) % 256) as u8;
            u32::from_be_bytes([a, b, c, d])
        })
        .collect()
}

#[cfg(feature = "proptest")]
pub mod strategies {
    //! Proptest strategies for generating routing tables and addresses.

    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating a valid IPv4 route (address masked to prefix length).
    pub fn ipv4_route_strategy() -> impl Strategy<Value = ([u8; 4], u8, u32)> {
        (any::<[u8; 4]>(), 0u8..=32, any::<u32>())
            .prop_map(|(addr, len, nexthop)| (mask_v4(addr, len), len, nexthop))
    }

    /// Strategy for generating a valid IPv6 route (address masked to prefix length).
    pub fn ipv6_route_strategy() -> impl Strategy<Value = ([u8; 16], u8, u128)> {
        (any::<[u8; 16]>(), 0u8..=128, any::<u128>())
            .prop_map(|(addr, len, nexthop)| (mask_v6(addr, len), len, nexthop))
    }

    /// Strategy for generating an IPv4 routing table with 1 to 8192 routes.
    pub fn ipv4_table_strategy() -> impl Strategy<Value = Ipv4RoutingTable<u32>> {
        prop::collection::vec(ipv4_route_strategy(), 1..8192).prop_map(|routes| {
            let mut table = Ipv4RoutingTable::default();
            for (addr, len, nexthop) in routes {
                table.add(addr, len, nexthop);
            }
            table
        })
    }

    /// Strategy for generating an IPv6 routing table with 1 to 8192 routes.
    pub fn ipv6_table_strategy() -> impl Strategy<Value = Ipv6RoutingTable<u128>> {
        prop::collection::vec(ipv6_route_strategy(), 1..8192).prop_map(|routes| {
            let mut table = Ipv6RoutingTable::default();
            for (addr, len, nexthop) in routes {
                table.add(addr, len, nexthop);
            }
            table
        })
    }
}
