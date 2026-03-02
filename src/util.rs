// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Copyright 2026 Oxide Computer Company

//! Display machinery

use crate::{
    Interior, IpAddress, IpRoutingTable, Ipv4RoutingTable, Ipv6RoutingTable,
    Poptrie,
};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::{Deref, DerefMut};

pub fn extract<Ip: IpAddress>(width: u8, offset: u8, v: Ip) -> u8 {
    let shift = Ip::BITS.saturating_sub(width * (offset + 1));
    let mask: Ip = Ip::from(0b111111) << shift;
    let res = (v & mask) >> shift;
    res.to_u8()
}

impl Debug for Interior {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut islots = Vec::new();
        let mut lslots = Vec::new();
        for i in 0..64 {
            if (self.iv & (1u64 << i)) != 0 {
                islots.push(i);
            }
        }
        for i in 0..64 {
            if (self.lv & (1u64 << i)) != 0 {
                lslots.push(i);
            }
        }
        //NOTE casts here due to
        //  - https://github.com/rust-lang/rust-analyzer/issues/11847
        f.debug_struct("Interior")
            .field("iv", &islots as &dyn core::fmt::Debug)
            .field("lv", &lslots as &dyn core::fmt::Debug)
            .field("interior_offset", &self.interior_offset)
            .field("leaf_offset", &self.leaf_offset)
            .finish()
    }
}

// NOTE #[derive(Default)] is broken see:
// https://github.com/rust-lang/rust/issues/26925
impl<T> Default for Poptrie<T> {
    fn default() -> Self {
        Self {
            interior: Vec::new(),
            leaf: Vec::new(),
            default: None,
        }
    }
}

// NOTE #[derive(Default)] is broken see:
// https://github.com/rust-lang/rust/issues/26925
impl<T> Default for Ipv4RoutingTable<T> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

// NOTE #[derive(Default)] is broken see:
// https://github.com/rust-lang/rust/issues/26925
impl<T> Default for Ipv6RoutingTable<T> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<Ip: IpAddress, T> Default for IpRoutingTable<Ip, T> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

/// Create a mask with bits 0 through n (inclusive) set.
/// For n=63, returns u64::MAX (all bits set).
#[inline]
pub fn mask_through(n: u8) -> u64 {
    if n >= 63 {
        u64::MAX
    } else {
        (1u64 << (n + 1)) - 1
    }
}

impl<T> Deref for Ipv4RoutingTable<T> {
    type Target = BTreeMap<([u8; 4], u8), T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Ipv4RoutingTable<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Deref for Ipv6RoutingTable<T> {
    type Target = BTreeMap<([u8; 16], u8), T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Ipv6RoutingTable<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
