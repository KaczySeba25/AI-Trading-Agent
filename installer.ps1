Write-Host 'Instalujê pe³ny system AI Trading Organism...'

# Usuwanie starych plików
 = @(
    'agent_rl.py',
    'environment.py',
    'train_agent_live.py',
    'run_agent.py',
    'evolution.py',
    'watchdog.ps1'
)

foreach ( in ) {
    if (Test-Path ) {
        Remove-Item  -Force
        Write-Host 'Usuniêto: ' + 
    }
}

# Tworzenie nowych plików
Set-Content feature_extractor_88.py 'PLACEHOLDER_FE'
Set-Content replay_buffer.py 'PLACEHOLDER_RB'
Set-Content environment_features.py 'PLACEHOLDER_ENV'
Set-Content agent_ppo_88.py 'PLACEHOLDER_AGENT'
Set-Content live_data.py 'PLACEHOLDER_LIVE'
Set-Content train_live_organism.py 'PLACEHOLDER_TRAIN'

Write-Host 'Instalacja zakoñczona.'
