# create_project_structure.ps1

# Root directory
$root = "algo_trading_project"

# Define folders
$folders = @(
    "$root\data\historical",
    "$root\indicators",
    "$root\models",
    "$root\strategy",
    "$root\utils"
)

# Define files
$files = @(
    "$root\main.py",
    "$root\config.py",
    "$root\requirements.txt",
    "$root\README.md",
    "$root\indicators\sma.py",
    "$root\indicators\ema.py",
    "$root\indicators\rsi.py",
    "$root\indicators\macd.py",
    "$root\models\ml_model.py",
    "$root\strategy\rule_based_strategy.py",
    "$root\utils\data_loader.py",
    "$root\utils\logger.py",
    "$root\utils\backtester.py"
)

# Create folders
foreach ($folder in $folders) {
    if (-Not (Test-Path -Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
    }
}

# Create files with placeholder content
foreach ($file in $files) {
    if (-Not (Test-Path -Path $file)) {
        New-Item -ItemType File -Path $file | Out-Null
        Add-Content -Path $file -Value "# Placeholder for $([System.IO.Path]::GetFileName($file))"
    }
}

Write-Host "Project structure created successfully under $root"
# End of script
# Save this script as create_project_structure.ps1 and run it in PowerShell to create the project structure.
# Ensure you have the necessary permissions to create directories and files in the specified location.