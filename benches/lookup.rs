use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use poptrie::Poptrie;
use poptrie_test_util::{generate_addrs, generate_table, longest_match_v4};

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup");

    for size in [100, 500, 1000, 5000, 10000] {
        let table = generate_table(size);
        let pt = Poptrie::from(table.clone());
        let addrs = generate_addrs(1000);

        group.bench_with_input(
            BenchmarkId::new("poptrie", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for addr in &addrs {
                        black_box(pt.match_v4(black_box(*addr)));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for addr in &addrs {
                        black_box(longest_match_v4(
                            &table,
                            black_box((*addr).to_be_bytes()),
                        ));
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    for size in [100, 500, 1000, 5000] {
        let table = generate_table(size);

        group.bench_with_input(
            BenchmarkId::new("poptrie", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(Poptrie::from(table.clone()));
                })
            },
        );
    }

    group.finish();
}

fn bench_single_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_lookup");

    for size in [100, 1000, 10000] {
        let table = generate_table(size);
        let pt = Poptrie::from(table.clone());
        let addr = u32::from_be_bytes([192, 168, 1, 1]);
        let addr_bytes = [192u8, 168, 1, 1];

        group.bench_with_input(
            BenchmarkId::new("poptrie", size),
            &size,
            |b, _| b.iter(|| black_box(pt.match_v4(black_box(addr)))),
        );

        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(longest_match_v4(
                        &table,
                        black_box(addr_bytes),
                    ))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lookup,
    bench_construction,
    bench_single_lookup
);
criterion_main!(benches);
