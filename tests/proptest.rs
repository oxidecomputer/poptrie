use poptrie::{Ipv4RoutingTable, Poptrie};
use poptrie_test_util::{
    longest_match_v4, longest_match_v6, mask_v4,
    strategies::{ipv4_table_strategy, ipv6_table_strategy},
};
use proptest::prelude::*;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::time::Instant;

proptest! {
    /// Test that looking up any address returns the same result as a naive
    /// longest-prefix-match implementation
    #[test]
    fn poptrie_matches_naive_lpm_v4(
        table in ipv4_table_strategy(),
        lookup_addr in any::<[u8; 4]>()
    ) {
        let routes = table.len();

        let t0 = Instant::now();
        let pt = Poptrie::from(table.clone());
        let construct_time = t0.elapsed();

        let addr_u32 = u32::from_be_bytes(lookup_addr);

        let t1 = Instant::now();
        let poptrie_result = pt.match_v4(addr_u32);
        let poptrie_time = t1.elapsed();

        let t2 = Instant::now();
        let naive_result = longest_match_v4(&table, lookup_addr);
        let naive_time = t2.elapsed();

        eprintln!("routes={routes:5} construct={construct_time:>10?} poptrie={poptrie_time:>10?} naive={naive_time:>10?}");

        prop_assert_eq!(poptrie_result, naive_result,
            "Mismatch for addr {:?}", Ipv4Addr::from(lookup_addr));
    }

    /// Test that looking up any address returns the same result as a naive
    /// longest-prefix-match implementation
    #[test]
    fn poptrie_matches_naive_lpm_v6(
        table in ipv6_table_strategy(),
        lookup_addr in any::<[u8; 16]>()
    ) {
        let pt = Poptrie::from(table.clone());
        let addr_u128 = u128::from_be_bytes(lookup_addr);

        let poptrie_result = pt.match_v6(addr_u128);
        let naive_result = longest_match_v6(&table, lookup_addr);

        prop_assert_eq!(poptrie_result, naive_result,
            "Mismatch for addr {:?}", Ipv6Addr::from(lookup_addr));
    }

    /// Test that every inserted route can be matched by an address within its prefix
    #[test]
    fn inserted_routes_are_matchable_v4(
        table in ipv4_table_strategy()
    ) {
        let pt = Poptrie::from(table.clone());

        for ((prefix_addr, prefix_len), _nexthop) in table.iter() {
            // Look up the prefix address itself
            let addr_u32 = u32::from_be_bytes(*prefix_addr);
            let result = pt.match_v4(addr_u32);

            // The result should exist and match either this route or a more specific one
            prop_assert!(result.is_some(),
                "No match for prefix {:?}/{}", Ipv4Addr::from(*prefix_addr), prefix_len);

            // Verify using naive LPM
            let naive_result = longest_match_v4(&table, *prefix_addr);
            prop_assert_eq!(result, naive_result);
        }
    }

    /// Test that every inserted route can be matched by an address within its prefix
    #[test]
    fn inserted_routes_are_matchable_v6(
        table in ipv6_table_strategy()
    ) {
        let pt = Poptrie::from(table.clone());

        for ((prefix_addr, prefix_len), _nexthop) in table.iter() {
            // Look up the prefix address itself
            let addr_u128 = u128::from_be_bytes(*prefix_addr);
            let result = pt.match_v6(addr_u128);

            // The result should exist and match either this route or a more specific one
            prop_assert!(result.is_some(),
                "No match for prefix {:?}/{}", Ipv6Addr::from(*prefix_addr), prefix_len);

            // Verify using naive LPM
            let naive_result = longest_match_v6(&table, *prefix_addr);
            prop_assert_eq!(result, naive_result);
        }
    }

    /// Test with routes that have long prefix lengths (/31 and /32) which
    /// exercise the edge cases at the final trie depth
    #[test]
    fn long_prefix_routes_v4(
        base_addr in any::<[u8; 3]>(),
        last_bytes in prop::collection::vec(any::<u8>(), 1..20),
        prefix_lens in prop::collection::vec(24u8..=32, 1..20),
    ) {
        let mut table = Ipv4RoutingTable::default();

        for (i, (last_byte, prefix_len)) in last_bytes.iter().zip(prefix_lens.iter()).enumerate() {
            let addr = [base_addr[0], base_addr[1], base_addr[2], *last_byte];
            let masked = mask_v4(addr, *prefix_len);
            table.add(masked, *prefix_len, i as u32);
        }

        let pt = Poptrie::from(table.clone());

        // Test lookups for each route
        for ((prefix_addr, prefix_len), _expected) in table.iter() {
            let addr_u32 = u32::from_be_bytes(*prefix_addr);
            let poptrie_result = pt.match_v4(addr_u32);
            let naive_result = longest_match_v4(&table, *prefix_addr);
            prop_assert_eq!(poptrie_result, naive_result,
                "Mismatch for {:?}/{}", Ipv4Addr::from(*prefix_addr), prefix_len);
        }
    }

    /// Test with dense routing tables where many routes share common prefixes
    /// (similar structure to the bug-triggering test case)
    #[test]
    fn dense_routes_v4(
        prefix_16 in any::<[u8; 2]>(),
        third_bytes in prop::collection::hash_set(any::<u8>(), 1..10),
        fourth_bytes in prop::collection::vec(any::<u8>(), 1..10),
    ) {
        let mut table = Ipv4RoutingTable::default();
        let mut nexthop = 0u32;

        // Add routes with varying third and fourth bytes
        for third in &third_bytes {
            for fourth in &fourth_bytes {
                // Mix of /24, /31, and /32 routes
                for prefix_len in [24u8, 31, 32] {
                    let addr = [prefix_16[0], prefix_16[1], *third, *fourth];
                    let masked = mask_v4(addr, prefix_len);
                    table.add(masked, prefix_len, nexthop);
                    nexthop += 1;
                }
            }
        }

        let pt = Poptrie::from(table.clone());

        // Test random lookups
        for third in &third_bytes {
            for fourth in 0u8..=255 {
                let addr = [prefix_16[0], prefix_16[1], *third, fourth];
                let addr_u32 = u32::from_be_bytes(addr);
                let poptrie_result = pt.match_v4(addr_u32);
                let naive_result = longest_match_v4(&table, addr);
                prop_assert_eq!(poptrie_result, naive_result,
                    "Mismatch for {:?}", Ipv4Addr::from(addr));
            }
        }
    }

    /// Test with routes that create sibling subtrees at various depths
    #[test]
    fn sibling_subtrees_v4(
        routes in prop::collection::vec(
            (any::<[u8; 4]>(), 8u8..=32, any::<u32>()),
            2..30
        )
    ) {
        let mut table = Ipv4RoutingTable::default();
        for (addr, len, nexthop) in &routes {
            let masked = mask_v4(*addr, *len);
            table.add(masked, *len, *nexthop);
        }

        let pt = Poptrie::from(table.clone());

        // Lookup each route's prefix address
        for (addr, len, _) in &routes {
            let masked = mask_v4(*addr, *len);
            let addr_u32 = u32::from_be_bytes(masked);
            let poptrie_result = pt.match_v4(addr_u32);
            let naive_result = longest_match_v4(&table, masked);
            prop_assert_eq!(poptrie_result, naive_result,
                "Mismatch for {:?}/{}", Ipv4Addr::from(masked), len);
        }

        // Also test with the original (unmasked) addresses
        for (addr, _len, _) in &routes {
            let addr_u32 = u32::from_be_bytes(*addr);
            let poptrie_result = pt.match_v4(addr_u32);
            let naive_result = longest_match_v4(&table, *addr);
            prop_assert_eq!(poptrie_result, naive_result,
                "Mismatch for {:?}", Ipv4Addr::from(*addr));
        }
    }
}

/// Specific regression test for the /31 and /32 bug pattern
#[test]
fn regression_31_32_pattern() {
    // This pattern mimics the original failing test
    let mut table = Ipv4RoutingTable::default();

    // Multiple /31 routes with same first two bytes but different third byte
    table.add([169, 254, 0, 0], 31, 1);
    table.add([169, 254, 0, 2], 31, 2);
    table.add([169, 254, 0, 6], 31, 3);

    // Multiple /32 routes with different third byte
    table.add([169, 254, 254, 1], 32, 7);
    table.add([169, 254, 254, 2], 32, 8);
    table.add([169, 254, 254, 4], 32, 9);

    let pt = Poptrie::from(table.clone());

    // Verify each /32 route
    for (addr, expected) in [
        ([169, 254, 254, 1], 7),
        ([169, 254, 254, 2], 8),
        ([169, 254, 254, 4], 9),
    ] {
        let addr_u32 = u32::from_be_bytes(addr);
        let result = pt.match_v4(addr_u32);
        assert_eq!(result, Some(expected), "Failed for {:?}", addr);
    }

    // Verify /31 routes
    for (addr, _expected) in [
        ([169, 254, 0, 0], 1),
        ([169, 254, 0, 1], 1), // Within /31
        ([169, 254, 0, 2], 2),
        ([169, 254, 0, 3], 2), // Within /31
        ([169, 254, 0, 6], 3),
        ([169, 254, 0, 7], 3), // Within /31
    ] {
        let addr_u32 = u32::from_be_bytes(addr);
        let result = pt.match_v4(addr_u32);
        let naive = longest_match_v4(&table, addr);
        assert_eq!(result, naive, "Failed for {:?}", addr);
    }
}
