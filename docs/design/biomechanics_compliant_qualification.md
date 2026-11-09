# Mixed compliant qualification row

Use the frozen rigid benchmark task, optimizer, three controller seeds, perturbation clocks, held-out bank, numerical tolerances and resource limits. Change the two biarticular muscle paths to the shared compliant Hill law. Preserve the other four rigid laws, all attachments, isometric force, optimal fiber length, tendon slack length, activation periods and initial activation. Use zero pennation for both benchmark profiles so the comparison varies tendon compliance.

The two compliant paths have tendon slack lengths of approximately 0.2774 m and 0.3378 m. Their state contains activation and fiber length. Start each trial at the actual reaching pose and equilibrate its fiber state at 0.05 activation. In float64, require projected isometric fiber force and spring tension at that pose to agree within 0.1 µN. Preserve full state on continuation.

Compare the matched analytical and MJX physical rows before performance timing. Use the same 10 ms controller clock, Euler starting at 1 ms physics, and step refinement to 0.5 ms and 0.25 ms. Use the coarsest step whose full loss, endpoints and directional derivatives satisfy the frozen limits. The force-balance check is at isometric equilibrium. The law bounds fiber speed during motion when spring tension exceeds attainable fiber force. Require held-out learning from the same fresh controller initializations as the rigid row.

Judge whether MJX meets the frozen practical limits before proposing further work on a native generic engine. A slow analytical-to-MJX ratio alone does not establish a need for a new engine. An unresolved numerical or learning failure remains a benchmark failure.
