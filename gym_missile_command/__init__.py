from gymnasium.envs.registration import register

register(
    id="missile-command-v0",
    entry_point="gym_missile_command.environment:MissileCommandEnv",
)
