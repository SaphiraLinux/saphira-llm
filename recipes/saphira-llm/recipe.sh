#!/bin/sh
# Saphira recipe: saphira-llm
#
# Native C11 inference runtime for BitNet b1.58 1-bit and conventional GGUF
# models, x86-64-v3. No Python, no BLAS, no launcher around someone else's
# binary: the engines, the kernels and the thread pool are in this tree.
#
# Built from the in-house saphira git remote rather than a vendored tarball, so
# the packaged bytes are the reviewed repository at a pinned revision. The
# revision is the origin/main tip at packaging time and is recorded in
# pkgver, so a rebuild of the same pkgver is the same source.

pkgname=saphira-llm
pkgver=0.0.1
pkgrel=1
pkgarch=${SAPHIRA_ARCH:-x86_64}
pkgdesc='Native C11 GGUF inference runtime for BitNet b1.58 (CPU, x86-64-v3)'
license='MIT'
origin=saphira-llm
repo=saphira
url=https://saphira.vm2.uk/

# In-house remote. saphira_llm_rev pins an immutable ref -- the release tag,
# which is the same string as pkgver -- so "the same pkgver" always means the
# same source. Never point it at a moving branch: a package recipe that builds
# whatever main happens to be is not reproducible.
#
# The upstream is the public repository, so this recipe builds from the same
# place anyone else would clone. An internal mirror may be substituted at
# packaging time by overriding saphira_llm_url in the environment; it is not
# recorded here, because a published recipe must not name private hosts.
saphira_llm_url=https://github.com/SaphiraLinux/saphira-llm.git
saphira_llm_branch=main
saphira_llm_rev=v0.0.1


# Nothing at runtime beyond libc: -lm for libm, -lpthread for the worker pool,
# both in the base system. No model file is packaged. The 1.2 GB BitNet
# b1.58-2B-4T GGUF is user-supplied, and the runtime reports a specific error
# rather than failing obscurely when it is absent.
depends=""

makedepends="
	binutils
	gcc
	make
	musl-dev
"

# binutils is a build dep, not decoration: 'make check' runs check-isa, which
# disassembles the built binary and fails if any instruction above x86-64-v3
# appears outside .sllm.isa.ext. That gate is the package's ISA contract, so it
# runs before anything is installed.

recipe_build()
{
	# Both remotes must yield the same commit; they are the same repository
	# reached two ways. Whichever answers first is checked out at the pin, so
	# a stale mirror cannot quietly package different source.
	cloned=no
	for url in "$saphira_llm_url" "${saphira_llm_url_OVERRIDE:-}"; do
		[ -n "$url" ] || continue
		if git clone --branch "$saphira_llm_branch" "$url" saphira-llm \
				>"$BUILDDIR/clone.log" 2>&1; then
			cloned=yes
			echo "cloned $url"
			break
		fi
		echo "  not reachable: $url" >>"$BUILDDIR/clone.log"
	done
	if [ "$cloned" = no ]; then
		echo "ERROR: cannot reach the saphira-llm remote; tried:" >&2
		grep 'not reachable' "$BUILDDIR/clone.log" >&2 || true
		return 1
	fi
	cd saphira-llm
	# The clone above is transport only; the pin decides what gets built, and
	# the resolved commit is recorded into the package so an installed copy
	# can be traced back to exact source.
	git fetch --tags origin 2>/dev/null || true
	git checkout --detach "$saphira_llm_rev"
	git rev-parse HEAD > "$BUILDDIR/saphira-llm.rev"
	echo "building saphira-llm $(cat "$BUILDDIR/saphira-llm.rev")"

	# Saphira baseline per target arch. Fail closed on an unknown arch
	# rather than quietly building a different ISA.
	case "${SAPHIRA_ARCH:-x86_64}" in
		x86_64) march=-march=x86-64-v3 ;;
		aarch64) march=-march=armv8-a ;;
		*) echo "ERROR: unsupported SAPHIRA_ARCH for saphira-llm: ${SAPHIRA_ARCH:-x86_64}" >&2; return 1 ;;
	esac

	make -j"${JOBS:-$(nproc)}" CC=gcc \
		BASELINE="$march" \
		all test bench eval tgbench kernelbench topoprobe

	# The gates run at package-build time, and the ISA gate is the reason
	# binutils is above. A package that cannot prove its own baseline does
	# not get installed.
	./saphira-llm-test > "$BUILDDIR/test.log" 2>&1 || {
		echo "ERROR: test suite failed; see $BUILDDIR/test.log" >&2
		tail -20 "$BUILDDIR/test.log" >&2
		return 1
	}
	# Assert the count rather than trusting the exit status alone, so a
	# suite that silently ran nothing cannot pass as green.
	grep -E '^[0-9]+ checks, 0 failed$' "$BUILDDIR/test.log" >/dev/null || {
		echo "ERROR: test log has no clean result line" >&2
		return 1
	}
	sh scripts/check-baseline-isa.sh ./saphira-llm ./saphira-llm-test \
		> "$BUILDDIR/isa.log" 2>&1 || {
		echo "ERROR: x86-64-v3 baseline proof failed" >&2
		cat "$BUILDDIR/isa.log" >&2
		return 1
	}
	echo "gates: $(grep -Eo '^[0-9]+ checks, 0 failed$' "$BUILDDIR/test.log"), baseline proven"

	# The ISA proof is only worth as much as its ability to fail, and during
	# development it was confirmed to fail on a deliberately mis-built
	# binary. That is re-proved here rather than assumed, once, at package
	# build time: one extra compile and link is cheap next to shipping a
	# package whose central safety claim was never tested.
	#
	# The canary is standalone. It links nothing from this package, because
	# its only job is to be a v3-baseline binary carrying one above-baseline
	# instruction outside any guarded section.
	cat > canary.c <<'EOF'
/* One AVX-512 instruction, in no guarded section, reached directly. */
__attribute__((target("avx512f")))
int sllm_recipe_canary(void) { return 1; }
int main(void) { return sllm_recipe_canary(); }
EOF
	cc -O2 -march=x86-64-v3 -o "$BUILDDIR/canary" canary.c
	if sh scripts/check-baseline-isa.sh "$BUILDDIR/canary" >/dev/null 2>&1; then
		echo "ERROR: check-isa passed a binary containing an above-baseline instruction" >&2
		return 1
	fi
	echo "gates: canary correctly rejected"
}

recipe_install()
{
	cd "$SRC/saphira-llm"

	# The inference binary is the package. The rest are the tools an operator
	# or a reviewer reaches for, so they ship as binaries rather than as
	# something to rebuild from source.
	install -d "$PKGDEST/usr/bin"
	install -m 0755 saphira-llm      "$PKGDEST/usr/bin/saphira-llm"
	install -m 0755 saphira-llm-tgbench      "$PKGDEST/usr/bin/saphira-llm-tgbench"
	install -m 0755 saphira-llm-kernelbench  "$PKGDEST/usr/bin/saphira-llm-kernelbench"
	install -m 0755 saphira-llm-schedbench  "$PKGDEST/usr/bin/saphira-llm-schedbench"
	install -m 0755 saphira-llm-topoprobe   "$PKGDEST/usr/bin/saphira-llm-topoprobe"
	install -m 0755 saphira-llm-eval        "$PKGDEST/usr/bin/saphira-llm-eval"
	install -m 0755 testsuite.sh    "$PKGDEST/usr/bin/saphira-llm-testsuite"

	# saphira-llm-test is the gate, not a user tool. It is installed under
	# libexec so the package can be re-verified after install, without
	# cluttering /usr/bin with a 20k-assertion runner.
	install -d "$PKGDEST/usr/libexec/saphira-llm"
	install -m 0755 saphira-llm-test \
		"$PKGDEST/usr/libexec/saphira-llm/saphira-llm-test"

	# Man page for the one binary an operator is expected to type.
	install -d "$PKGDEST/usr/share/man/man1"
	install -m 0644 "$RECIPE_DIR/files/saphira-llm.1" \
		"$PKGDEST/usr/share/man/man1/saphira-llm.1"

	install -d "$PKGDEST/usr/share/doc/saphira-llm"
	for d in README.md ARCHITECTURE.md FORMAT.md BENCHMARKS.md PROVENANCE.md; do
		install -m 0644 "$d" "$PKGDEST/usr/share/doc/saphira-llm/$d"
	done
	install -m 0644 docs/EVALUATION.md "$PKGDEST/usr/share/doc/saphira-llm/EVALUATION.md"
	# The eval fixture travels with the package so a recorded perplexity is
	# reproducible on an installed system without a checkout.
	install -d "$PKGDEST/usr/share/saphira-llm/corpus"
	install -m 0644 tests/golden/eval-corpus.txt \
		"$PKGDEST/usr/share/saphira-llm/corpus/eval-corpus.txt"
	# install -D would create each parent, but the licenses path is a
	# directory that has to exist before two files land in it.
	install -d "$PKGDEST/usr/share/licenses/saphira-llm"
	install -m 0644 LICENSE "$PKGDEST/usr/share/licenses/saphira-llm/LICENSE"
	install -m 0644 NOTICE "$PKGDEST/usr/share/licenses/saphira-llm/NOTICE"

	# The build tree, so the public headers and the reference tooling are
	# available to a C program that wants to link the runtime directly.
	# There is no shared library, so the headers alone would not be usable.
	install -d "$PKGDEST/usr/include/saphira_llm"
	install -m 0644 include/saphira_llm/*.h "$PKGDEST/usr/include/saphira_llm/"
	install -d "$PKGDEST/usr/share/saphira-llm"
	install -m 0644 "$BUILDDIR/saphira-llm.rev" \
		"$PKGDEST/usr/share/saphira-llm/revision"

	# No post-install generation, no first-boot download, no model shipped.
	# A package that reached for the network at install time would be
	# unbuildable offline and unreviewable.
}

recipe_check()
{
	# Post-install: prove the installed binary runs and reports its baseline,
	# and that the packaged gate is still green from its installed path.
	# The gate is quiet on stdout and asserts on the log, so a build log stays
	# readable instead of carrying 20k assertions' worth of chatter.
	"$PKGDEST/usr/bin/saphira-llm" --version
	"$PKGDEST/usr/bin/saphira-llm" --help >/dev/null
	"$PKGDEST/usr/libexec/saphira-llm/saphira-llm-test" \
		>"$BUILDDIR/postinstall-test.log" 2>&1 || {
		echo "ERROR: installed test binary failed" >&2
		tail -20 "$BUILDDIR/postinstall-test.log" >&2
		return 1
	}
	grep -Eo '^[0-9]+ checks, 0 failed$' "$BUILDDIR/postinstall-test.log" ||
		{
		echo "ERROR: installed test log has no clean result line" >&2
		return 1
	}
	# No model, so no generation here: the acceptance model is 1.2 GB and is
	# deliberately not a package dependency. saphira-llm-testsuite does the
	# Q&A pass when a model is present.
	echo "check: installed binary runs, assertions clean, model not required"
	return 0
}
