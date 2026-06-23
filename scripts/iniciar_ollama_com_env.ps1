# Inicia o servidor Ollama com as variáveis OLLAMA_* definidas no .env do projeto.
$arquivoEnv = Join-Path $PSScriptRoot "..\.env"

if (-not (Test-Path -LiteralPath $arquivoEnv)) {
    throw "Arquivo .env nao encontrado em $PSScriptRoot"
}

Get-Content -LiteralPath $arquivoEnv | ForEach-Object {
    $linha = $_.Trim()
    if ($linha -match '^(OLLAMA_[A-Z0-9_]+)\s*=\s*(.*)$') {
        $nome = $matches[1]
        $valor = $matches[2].Trim().Trim('"').Trim("'")
        Set-Item -Path "Env:$nome" -Value $valor
        Write-Host "Configurado $nome"
    }
}

Write-Host "Iniciando Ollama com a configuracao de .env..."
& ollama serve
