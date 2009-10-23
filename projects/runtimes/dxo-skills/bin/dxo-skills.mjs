#!/usr/bin/env node

const command = process.argv[2] ?? 'help';

if (command === 'prompt') {
  console.log(`You are working on DXO Framework. First read AGENTS.md, then read the relevant design documents and roadmap entries.
Task: [one concrete outcome]
Scope: [files or package, if known]
Constraints: preserve the Rust/napi boundary, do not invent APIs, and keep preview/placeholder status honest.
Verification: run [specific pnpm verify gate or test].
Delivery: make the smallest reviewable change, show the diff, and report unrelated pre-existing changes without including them.`);
} else {
  console.log('DXO Skills — agent-first onboarding');
  console.log('Usage: npx @dxo/skills prompt');
}
