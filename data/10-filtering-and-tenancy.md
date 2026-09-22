# Metadata Filtering and Multi-Tenancy

Most real retrieval systems must restrict what a query can see: to one customer, one
date range, one document set, one permission level. How that restriction is applied
changes both correctness and performance.

## Pre-filter versus post-filter

A post-filter retrieves the top k by similarity and then discards anything that fails
the predicate. It is trivial to bolt on and it is usually wrong. If a tenant owns 1% of
the corpus, retrieving the global top 10 and filtering leaves roughly zero results, and
the failure is silent: a thin or empty result set with no error.

A pre-filter restricts the candidate set before the similarity search runs, so the top k
is computed over eligible documents only. This is correct by construction. It is harder
to implement efficiently, because an approximate index built over the whole corpus
cannot simply skip ineligible nodes without degrading its graph traversal.

The practical rule: post-filtering is acceptable only when the predicate is weakly
selective and keeps most of the corpus. Anything more selective needs a pre-filter, and
a highly selective predicate usually needs a separate index per partition.

## Selectivity

Selectivity is the fraction of the corpus a predicate keeps, and it governs the whole
design. A predicate keeping 80% of documents costs almost nothing and post-filtering is
fine. A predicate keeping 0.1% makes an approximate index nearly useless, because the
search walks a graph overwhelmingly composed of ineligible nodes; at that point an exact
scan over the eligible subset is both simpler and faster.

## Tenant isolation

Three arrangements, in increasing order of isolation and cost.

A shared index with a tenant field on every chunk is cheapest and scales to many small
tenants, but a filtering bug leaks one tenant's documents into another's answers, which
is the worst failure this kind of system can have.

A namespace or collection per tenant gives real isolation and keeps per-tenant indexes
small and fast. It costs memory per namespace, which becomes the binding constraint
somewhere in the low thousands of tenants.

A separate store per tenant is fully isolated and operationally heavy. It suits a small
number of large tenants with compliance requirements.

## Denormalise the filter fields

Whatever the arrangement, the fields used for filtering must live on the chunk itself.
Resolving a permission or an owner by joining to another system at query time puts a
network round trip inside the retrieval path, and it means an outage in that system
becomes an outage in search. Denormalise the fields at ingest and re-ingest when they
change.
