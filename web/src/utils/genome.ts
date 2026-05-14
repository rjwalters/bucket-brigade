import type { ArchetypeParams } from '../data/archetypes.generated';

// Convert genome array to ArchetypeParams object
// Genome order: [honesty_bias, work_tendency, neighbor_help_bias, own_house_priority,
//                risk_aversion, coordination_weight, exploration_rate, fatigue_memory,
//                rest_reward_bias, altruism_factor]
export function genomeToParams(genome: number[]): ArchetypeParams {
  return {
    honesty_bias: genome[0],
    work_tendency: genome[1],
    neighbor_help_bias: genome[2],
    own_house_priority: genome[3],
    risk_aversion: genome[4],
    coordination_weight: genome[5],
    exploration_rate: genome[6],
    fatigue_memory: genome[7],
    rest_reward_bias: genome[8],
    altruism_factor: genome[9],
  };
}
