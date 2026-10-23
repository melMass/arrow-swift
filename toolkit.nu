# arrow-swift dev commands: `overlay use toolkit.nu`

const root = path self .
const manifest = "apps/Quiver/app.yaml"

# Cut a release, all local: bump Quiver's version in apps/Quiver/app.yaml,
# regenerate CHANGELOG.md with git-cliff, then make one commit and one annotated
# tag that both carry the release notes (the changelog's new section with the
# Internal group hidden). Nothing is pushed: `git push --follow-tags` when
# ready, then `swiftkit dist Quiver` for the signed build.
export def release [
  bump: string # patch | minor | major | <x.y.z>
  --dry-run # print the version, the notes and the files it would touch; write nothing
] {
  cd $root
  if (^git status --porcelain | str trim | is-not-empty) {
    error make {msg: "uncommitted changes: commit or stash them first"}
  }
  let current = (open $manifest | get version)
  let next = (next-version $current $bump)
  let tag = $"v($next)"
  if (^git tag --list $tag | str trim | is-not-empty) {
    error make {msg: $"($tag) already exists"}
  }
  let text = (open --raw $manifest)
  let pattern = '(?m)^version: "[^"]*"$'
  if ($text | parse --regex $pattern | length) != 1 {
    error make {msg: $"expected exactly one `version:` line in ($manifest)"}
  }
  # The commits since the last tag, rendered as `tag` would be.
  let notes = (
    with-env {CLIFF_HIDE_INTERNAL: "1"} {
      ^git cliff --offline --unreleased --tag $tag --strip header
    } | str trim
  )
  let subject = $"chore\(release\): ($tag)"

  print $"(ansi gb)($current) → ($next)(ansi reset)  commit `($subject)`, tag ($tag)"
  print $"(ansi d)($manifest), CHANGELOG.md(ansi reset)\n"
  print $notes
  if $dry_run { return }

  $text | str replace --regex $pattern $'version: "($next)"' | save --force $manifest
  ^git cliff --offline --tag $tag --output CHANGELOG.md
  ^git add $manifest CHANGELOG.md
  ^git commit --quiet -m $subject -m $notes
  # Verbatim: the default cleanup drops the notes' `##` lines as comments.
  ^git tag --annotate --cleanup=verbatim $tag -m $notes
  print $"\n(ansi gb)Tagged ($tag)(ansi reset); push when ready: git push --follow-tags"
}

# `bump` applied to `version`: a level, or an explicit x.y.z.
def next-version [version: string, bump: string]: nothing -> string {
  let parts = ($version | split row "." | into int)
  match $bump {
    "major" => $"($parts.0 + 1).0.0"
    "minor" => $"($parts.0).($parts.1 + 1).0"
    "patch" => $"($parts.0).($parts.1).($parts.2 + 1)"
    _ => {
      if ($bump !~ '^\d+\.\d+\.\d+$') {
        error make {msg: $"expected patch, minor, major or x.y.z, got ($bump)"}
      }
      $bump
    }
  }
}
