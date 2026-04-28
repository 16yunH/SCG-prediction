param(
  [string]$Branch = "main",
  [string]$Remote = "https://github.com/16yunH/SCG-prediction.git"
)

if (-not (Test-Path .git)) {
  git init
}

$current = git branch --show-current
if (-not $current) {
  git checkout -b $Branch
} elseif ($current -ne $Branch) {
  git checkout -B $Branch
}

$hasOrigin = git remote | Select-String -Pattern "^origin$" -Quiet
if (-not $hasOrigin) {
  git remote add origin $Remote
} else {
  git remote set-url origin $Remote
}

git add .
git commit -m "Initial SCG BP pipeline and remote training scripts" --allow-empty
git push -u origin $Branch
