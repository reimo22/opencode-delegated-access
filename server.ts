// V2 directory-package entrypoint. The V2 loader (Host.resolve) only looks
// for <root>/server.* or <root>/index.* module files for local directory
// plugins — package.json "main"/"exports" are ignored there. This shim keeps
// the real implementation in src/ while satisfying that convention.
export { default } from "./src/index.ts"
