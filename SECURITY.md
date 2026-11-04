# Security

## Reporting a vulnerability in LLMInspect

Please report it privately, not in a public issue: open the repository's **Security** tab and choose
**Report a vulnerability**. Include what you ran, what happened and, if you can, a minimal model or file
that shows it. You'll get a reply within a week.

What counts:

- a file LLMInspect reads that makes it run code, write outside its data folder or read outside the
  folder it was pointed at;
- a way past public mode's limits (local paths, gated repos, rate limits, size caps);
- a token sent anywhere other than the Hugging Face Hub;
- a check that can be made to report nothing on a file it should flag, in a way you can reproduce.

## Findings about other people's models

LLMInspect flags models on the Hub whose files run code on load, claim a false parent or carry a targeted
weight edit. A finding is evidence, not a verdict. If you find a model that looks malicious, report it to
the hosting site (on Hugging Face, the model page's **Report** button) and to its author; please don't open
issues here naming third-party models.

## Supported versions

Only the latest commit on `main` gets fixes.
