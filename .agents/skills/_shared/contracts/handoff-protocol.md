# Skill Handoff Protocol

Every handoff records:

- `from_skill` and `to_skill`;
- input artifact paths and versions;
- completed checks;
- unresolved `unknown` or `conflict` fields;
- blocking Evidence Gap;
- requested next action;
- files that must remain read-only.

The receiving Skill must validate required inputs before work. It must not silently repair missing provenance or perform the sending Skill's responsibility.
