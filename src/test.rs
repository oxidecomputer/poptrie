// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Copyright 2026 Oxide Computer Company

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

fn extract_32(width: u8, offset: u8, v: u32) -> u8 {
    extract!(width, offset, v, 32u8)
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
fn test_match_v4_sus() {
    let tbl = test_routing_table_sus_v4();
    let pt = Poptrie::<Ipv4>::from(tbl);

    let addr = Ipv4::new([169, 254, 254, 1]);
    let m = pt.match_v4(addr.0);
    assert_eq!(m, Some(Ipv4::new([1, 1, 1, 7])));
}

fn test_routing_table_sus_v4() -> Ipv4RoutingTable<Ipv4> {
    let mut tbl = Ipv4RoutingTable::<Ipv4>::default();

    tbl.add([169, 254, 0, 0], 31, Ipv4::new([1, 1, 1, 1]));
    tbl.add([169, 254, 0, 2], 31, Ipv4::new([1, 1, 1, 2]));
    tbl.add([169, 254, 0, 6], 31, Ipv4::new([1, 1, 1, 3]));
    tbl.add([169, 254, 0, 32], 31, Ipv4::new([1, 1, 1, 4]));
    tbl.add([169, 254, 0, 34], 31, Ipv4::new([1, 1, 1, 5]));
    tbl.add([169, 254, 0, 38], 31, Ipv4::new([1, 1, 1, 6]));
    tbl.add([169, 254, 254, 1], 32, Ipv4::new([1, 1, 1, 7]));
    tbl.add([169, 254, 254, 2], 32, Ipv4::new([1, 1, 1, 8]));
    tbl.add([169, 254, 254, 4], 32, Ipv4::new([1, 1, 1, 9]));
    tbl.add([172, 20, 29, 0], 24, Ipv4::new([1, 1, 1, 10]));
    tbl
}

#[test]
fn test_match_v4_multi() {
    let tbl = test_routing_table_v4_mp();
    let pt = Poptrie::<Vec<Ipv4>>::from(tbl);

    // Test hits
    let addr = Ipv4::new([1, 7, 0, 1]);
    let m = pt.match_v4(addr.0);
    assert_eq!(
        m,
        Some(vec![
            Ipv4::new([1, 254, 254, 254]), // path 1
            Ipv4::new([1, 254, 254, 255]), // path 2
        ])
    );

    let addr = Ipv4::new([247, 33, 0, 1]);
    let m = pt.match_v4(addr.0);
    assert_eq!(
        m,
        Some(vec![
            Ipv4::new([247, 33, 0, 1]), // path 1
            Ipv4::new([247, 33, 0, 2]), // path 2
        ])
    );

    let addr = Ipv4::new([247, 33, 12, 1]);
    let m = pt.match_v4(addr.0);
    assert_eq!(
        m,
        Some(vec![
            Ipv4::new([247, 33, 12, 1]), // path 1
            Ipv4::new([247, 33, 12, 2]), // path 2
        ])
    );

    let addr = Ipv4::new([51, 12, 109, 1]);
    let m = pt.match_v4(addr.0);
    assert_eq!(
        m,
        Some(vec![
            Ipv4::new([51, 12, 109, 10]), // path 1
            Ipv4::new([51, 12, 109, 11]), // path 2
        ],)
    );

    let addr = Ipv4::new([77, 18, 4, 7]);
    let m = pt.match_v4(addr.0);
    assert_eq!(
        m,
        Some(vec![
            Ipv4::new([77, 18, 10, 1]), // path 1
            Ipv4::new([77, 18, 10, 2]), // path 2
        ],)
    );

    let addr = Ipv4::new([170, 1, 14, 3]);
    let m = pt.match_v4(addr.0);
    assert_eq!(
        m,
        Some(vec![
            Ipv4::new([1, 7, 0, 1]), // path 1
            Ipv4::new([1, 7, 0, 2]), // path 2
        ],)
    );

    // Test default route
    let addr = Ipv4::new([4, 7, 0, 1]);
    let m = pt.match_v4(addr.0);
    assert_eq!(m, None);

    let tbl = test_routing_table_with_default_route_v4_mp();
    let pt = Poptrie::<Vec<Ipv4>>::from(tbl);

    // Test default route
    let addr = Ipv4::new([4, 7, 0, 1]);
    let m = pt.match_v4(addr.0);
    assert_eq!(
        m,
        Some(vec![
            Ipv4::new([1, 2, 3, 4]), // path 1
            Ipv4::new([1, 2, 3, 5]), // path 2
        ])
    );
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

fn test_routing_table_v4_mp() -> Ipv4RoutingTable<Vec<Ipv4>> {
    let mut tbl = Ipv4RoutingTable::<Vec<Ipv4>>::default();
    tbl.add(
        [1, 0, 0, 0],
        8,
        vec![
            Ipv4::new([1, 254, 254, 254]), // path 1
            Ipv4::new([1, 254, 254, 255]), // path 2
        ],
    );

    tbl.add(
        [247, 33, 0, 0],
        16,
        vec![
            Ipv4::new([247, 33, 0, 1]), // path 1
            Ipv4::new([247, 33, 0, 2]), // path 2
        ],
    );

    tbl.add(
        [247, 33, 12, 0],
        24,
        vec![
            Ipv4::new([247, 33, 12, 1]), // path 1
            Ipv4::new([247, 33, 12, 2]), // path 2
        ],
    );

    tbl.add(
        [51, 12, 109, 0],
        24,
        vec![
            Ipv4::new([51, 12, 109, 10]), // path 1
            Ipv4::new([51, 12, 109, 11]), // path 2
        ],
    );

    tbl.add(
        [77, 18, 0, 0],
        16,
        vec![
            Ipv4::new([77, 18, 10, 1]), // path 1
            Ipv4::new([77, 18, 10, 2]), // path 2
        ],
    );

    tbl.add(
        [170, 1, 14, 3],
        32,
        vec![
            Ipv4::new([1, 7, 0, 1]), // path 1
            Ipv4::new([1, 7, 0, 2]), // path 2
        ],
    );

    tbl
}

fn test_routing_table_with_default_route_v4() -> Ipv4RoutingTable<Ipv4> {
    let mut tbl = test_routing_table_v4();
    tbl.add([0, 0, 0, 0], 0, Ipv4::new([1, 2, 3, 4]));
    tbl
}

fn test_routing_table_with_default_route_v4_mp() -> Ipv4RoutingTable<Vec<Ipv4>>
{
    let mut tbl = test_routing_table_v4_mp();
    tbl.add(
        [0, 0, 0, 0],
        0,
        vec![
            Ipv4::new([1, 2, 3, 4]), // path 1
            Ipv4::new([1, 2, 3, 5]), // path 2
        ],
    );
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

// Test case for underflow bug found by proptest
// Minimal failing input: routes [0.0.0.128/25, 0.0.0.0/8]
// (The proptest output showed [0.0.0.252/25] but that gets masked to [0.0.0.128/25])
#[test]
fn test_underflow_bug() {
    let mut tbl = Ipv4RoutingTable::<u32>::default();
    // Use properly masked addresses (host bits must be zero)
    tbl.add([0, 0, 0, 128], 25, 1); // 0.0.0.128/25
    tbl.add([0, 0, 0, 0], 8, 2); // 0.0.0.0/8

    let pt = Poptrie::from(tbl);

    // Debug output
    println!("interior nodes: {}", pt.interior.len());
    println!("leaf nodes: {}", pt.leaf.len());
    for (i, interior) in pt.interior.iter().enumerate() {
        println!(
            "  interior[{}]: ioff={}, loff={}",
            i, interior.interior_offset, interior.leaf_offset
        );
        println!("{:#?}", interior);
    }

    // Looking up 0.0.0.0 should match the /8 route
    let addr = u32::from_be_bytes([0, 0, 0, 0]);
    let result = pt.match_v4(addr);
    assert_eq!(result, Some(2));

    // Looking up 0.0.0.128 should match the /25 route
    let addr = u32::from_be_bytes([0, 0, 0, 128]);
    let result = pt.match_v4(addr);
    assert_eq!(result, Some(1));
}

// Test case for LPM bug found by proptest
// Minimal failing input: /1 and /2 prefixes at same address
// The more specific /2 should win over /1
#[test]
fn test_lpm_overlapping_prefixes() {
    let mut tbl = Ipv4RoutingTable::<u32>::default();
    tbl.add([0, 0, 0, 0], 1, 1); // /1 -> nexthop 1
    tbl.add([0, 0, 0, 0], 2, 0); // /2 -> nexthop 0 (more specific)

    let pt = Poptrie::from(tbl);

    // Debug output
    println!("interior nodes: {}", pt.interior.len());
    println!("leaf nodes: {}", pt.leaf.len());
    for (i, interior) in pt.interior.iter().enumerate() {
        println!(
            "  interior[{}]: ioff={}, loff={}",
            i, interior.interior_offset, interior.leaf_offset
        );
        println!("{:#?}", interior);
    }
    for (i, leaf) in pt.leaf.iter().enumerate() {
        println!("  leaf[{}]: {}", i, leaf.data);
    }

    // Looking up 0.0.0.0 should match the /2 route (more specific)
    let addr = u32::from_be_bytes([0, 0, 0, 0]);
    let result = pt.match_v4(addr);
    assert_eq!(
        result,
        Some(0),
        "Should match /2 (nexthop 0), not /1 (nexthop 1)"
    );
}

// Test case from proptest: short prefix (/1) should match addresses
// in its range even when there's a more specific route elsewhere.
#[test]
fn test_short_prefix_bug() {
    let mut tbl = Ipv4RoutingTable::<u32>::default();
    tbl.add([128, 0, 0, 0], 1, 1); // /1 -> nexthop 1 (covers 128.0.0.0 - 255.255.255.255)
    tbl.add([228, 0, 0, 0], 7, 2); // /7 -> nexthop 2

    let pt = Poptrie::from(tbl.clone());

    // Debug output
    println!("interior nodes: {}", pt.interior.len());
    println!("leaf nodes: {}", pt.leaf.len());
    for (i, interior) in pt.interior.iter().enumerate() {
        println!(
            "  interior[{}]: iv={:#018x}, lv={:#018x}, ioff={}, loff={}",
            i,
            interior.iv,
            interior.lv,
            interior.interior_offset,
            interior.leaf_offset
        );
    }
    for (i, leaf) in pt.leaf.iter().enumerate() {
        println!("  leaf[{}]: {}", i, leaf.data);
    }

    // Looking up 230.0.0.0 should match the /1 route
    // 230 = 0xE6, first 6 bits = 57
    let addr = u32::from_be_bytes([230, 0, 0, 0]);
    let result = pt.match_v4(addr);
    assert_eq!(result, Some(1), "Should match /1 (nexthop 1)");

    // Looking up 228.0.0.0 should match the /7 route
    let addr = u32::from_be_bytes([228, 0, 0, 0]);
    let result = pt.match_v4(addr);
    assert_eq!(result, Some(2), "Should match /7 (nexthop 2)");

    // Looking up 128.0.0.0 should match the /1 route
    let addr = u32::from_be_bytes([128, 0, 0, 0]);
    let result = pt.match_v4(addr);
    assert_eq!(result, Some(1), "Should match /1 (nexthop 1)");
}

// Test case from proptest minimal failing input:
// prefix_16 = [0, 0], third_bytes = {0}, fourth_bytes = [62]
// Routes: [0,0,0,0]/24, [0,0,0,62]/31, [0,0,0,62]/32
#[test]
fn test_dense_route_bug() {
    let mut tbl = Ipv4RoutingTable::<u32>::default();
    // Add routes with varying prefix lengths at same location
    // Note: /24 masks to [0,0,0,0], /31 and /32 stay as [0,0,0,62]
    tbl.add([0, 0, 0, 0], 24, 0); // nexthop 0 for /24
    tbl.add([0, 0, 0, 62], 31, 1); // nexthop 1 for /31
    tbl.add([0, 0, 0, 62], 32, 2); // nexthop 2 for /32

    let pt = Poptrie::from(tbl);

    // Debug output
    println!("interior nodes: {}", pt.interior.len());
    println!("leaf nodes: {}", pt.leaf.len());
    for (i, interior) in pt.interior.iter().enumerate() {
        println!(
            "  interior[{}]: iv={:#018x}, lv={:#018x}, ioff={}, loff={}",
            i,
            interior.iv,
            interior.lv,
            interior.interior_offset,
            interior.leaf_offset
        );
    }
    for (i, leaf) in pt.leaf.iter().enumerate() {
        println!("  leaf[{}]: {}", i, leaf.data);
    }

    // Looking up 0.0.0.62 should match the /32 route (most specific)
    let addr = u32::from_be_bytes([0, 0, 0, 62]);
    let result = pt.match_v4(addr);
    assert_eq!(result, Some(2), "Should match /32 (nexthop 2)");

    // Looking up 0.0.0.63 should match the /31 route
    let addr = u32::from_be_bytes([0, 0, 0, 63]);
    let result = pt.match_v4(addr);
    assert_eq!(result, Some(1), "Should match /31 (nexthop 1)");

    // Looking up 0.0.0.0 should match the /24 route
    let addr = u32::from_be_bytes([0, 0, 0, 0]);
    let result = pt.match_v4(addr);
    assert_eq!(result, Some(0), "Should match /24 (nexthop 0)");
}
