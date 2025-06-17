#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cstdint>

// Função hash original (problemática)
int prune_hash_original(int id_val, float prob) {
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;
    if (id_val == 0) return 1; // Thread 0 does not prune
    unsigned int hash_val = id_val * 1103515245U + 12345U;
    hash_val ^= hash_val >> 16;
    return ((hash_val & 0x7FFFFFFF) < (unsigned int)(prob * 2147483647.0f)) ? 1 : 0;
}

// Função hash melhorada - Opção 1: Hash mais robusto
int prune_hash_v1(int id_val, float prob) {
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;
    if (id_val == 0) return 1; // Thread 0 does not prune
    
    // MurmurHash3-like mixing
    uint32_t hash_val = (uint32_t)id_val;
    hash_val ^= hash_val >> 16;
    hash_val *= 0x85ebca6b;
    hash_val ^= hash_val >> 13;
    hash_val *= 0xc2b2ae35;
    hash_val ^= hash_val >> 16;
    
    return ((hash_val & 0x7FFFFFFF) < (uint32_t)(prob * 2147483647.0f)) ? 1 : 0;
}

// Função hash melhorada - Opção 2: Considera prob no hash
int prune_hash_v2(int id_val, float prob) {
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;
    if (id_val == 0) return 1; // Thread 0 does not prune
    
    // Incorpora a probabilidade no seed do hash
    uint32_t prob_bits = *(uint32_t*)&prob; // Reinterpreta float como uint32
    uint32_t seed = prob_bits ^ 0x9e3779b9;
    
    uint32_t hash_val = (uint32_t)id_val;
    hash_val ^= seed;
    hash_val ^= hash_val >> 16;
    hash_val *= 0x85ebca6b;
    hash_val ^= hash_val >> 13;
    hash_val *= 0xc2b2ae35;
    hash_val ^= hash_val >> 16;
    
    return ((hash_val & 0x7FFFFFFF) < (uint32_t)(prob * 2147483647.0f)) ? 1 : 0;
}

// Função hash melhorada - Opção 3: Hash específico por probabilidade
int prune_hash_v3(int id_val, float prob) {
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;
    if (id_val == 0) return 1; // Thread 0 does not prune
    
    // Cria um hash único combinando ID e índice da probabilidade
    // Assumindo que as probabilidades vêm em ordem fixa
    static int prob_counter = 0;
    static float last_prob = -1.0f;
    
    if (prob != last_prob) {
        prob_counter++;
        last_prob = prob;
    }
    
    uint64_t combined = ((uint64_t)id_val << 32) | (uint32_t)prob_counter;
    
    // Hash de 64-bit para 32-bit
    uint32_t hash_val = (uint32_t)(combined ^ (combined >> 32));
    hash_val ^= hash_val >> 16;
    hash_val *= 0x85ebca6b;
    hash_val ^= hash_val >> 13;
    hash_val *= 0xc2b2ae35;
    hash_val ^= hash_val >> 16;
    
    return ((hash_val & 0x7FFFFFFF) < (uint32_t)(prob * 2147483647.0f)) ? 1 : 0;
}

// Função hash melhorada - Opção 4: xxHash-like (mais robusta)
int prune_hash_v4(int id_val, float prob) {
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;
    if (id_val == 0) return 1; // Thread 0 does not prune
    
    // xxHash32-inspired
    const uint32_t PRIME32_1 = 0x9E3779B1;
    const uint32_t PRIME32_2 = 0x85EBCA77;
    const uint32_t PRIME32_3 = 0xC2B2AE3D;
    const uint32_t PRIME32_4 = 0x27D4EB2F;
    const uint32_t PRIME32_5 = 0x165667B1;
    
    uint32_t h32 = PRIME32_5 + 4; // seed + len
    h32 += (uint32_t)id_val * PRIME32_3;
    h32 = ((h32 << 17) | (h32 >> 15)) * PRIME32_4;
    
    h32 ^= h32 >> 15;
    h32 *= PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= PRIME32_3;
    h32 ^= h32 >> 16;
    
    return ((h32 & 0x7FFFFFFF) < (uint32_t)(prob * 2147483647.0f)) ? 1 : 0;
}

std::vector<float> readProbabilities(const std::string& filename) {
    std::vector<float> probabilities;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo: " << filename << std::endl;
        return probabilities;
    }
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            try {
                float prob = std::stof(line);
                probabilities.push_back(prob);
            } catch (const std::exception& e) {
                std::cerr << "Erro ao converter linha: " << line << std::endl;
            }
        }
    }
    
    file.close();
    return probabilities;
}

std::string vectorToString(const std::vector<int>& vec) {
    std::string result;
    result.reserve(vec.size());
    for (int val : vec) {
        result += (val == 1) ? '1' : '0';
    }
    return result;
}

struct HashTestResult {
    std::string name;
    int unique_patterns;
    float uniqueness_rate;
    int max_collision;
    std::map<std::string, std::vector<int>> patterns;
};

HashTestResult testHashFunction(const std::string& name, 
                               int (*hash_func)(int, float),
                               const std::vector<float>& probabilities,
                               int num_ids) {
    HashTestResult result;
    result.name = name;
    
    std::map<std::string, std::vector<int>> patternCount;
    
    for (int id = 0; id < num_ids; id++) {
        std::vector<int> pruneVector;
        pruneVector.reserve(probabilities.size());
        
        for (float prob : probabilities) {
            pruneVector.push_back(hash_func(id, prob));
        }
        
        std::string pattern = vectorToString(pruneVector);
        patternCount[pattern].push_back(id);
    }
    
    result.patterns = patternCount;
    result.unique_patterns = patternCount.size();
    result.uniqueness_rate = (float)result.unique_patterns / num_ids;
    
    // Encontra a maior colisão
    result.max_collision = 0;
    for (const auto& entry : patternCount) {
        result.max_collision = std::max(result.max_collision, (int)entry.second.size());
    }
    
    return result;
}

int main() {
    const int NUM_IDS = 10000;
    const std::string filename = "prune.csv";
    const float TARGET_UNIQUENESS = 0.90f; // 90%
    
    std::vector<float> probabilities = readProbabilities(filename);
    
    if (probabilities.empty()) {
        std::cerr << "Nenhuma probabilidade foi lida do arquivo!" << std::endl;
        return 1;
    }
    
    std::cout << "Probabilidades lidas: " << probabilities.size() << std::endl;
    std::cout << "Meta: " << (TARGET_UNIQUENESS * 100) << "% de padrões únicos (" 
              << (int)(TARGET_UNIQUENESS * NUM_IDS) << "+ padrões únicos)" << std::endl;
    std::cout << "\nTestando diferentes implementações de hash...\n" << std::endl;
    
    // Testa todas as versões
    std::vector<HashTestResult> results;
    
    std::cout << "Testando hash original..." << std::endl;
    results.push_back(testHashFunction("Original", prune_hash_original, probabilities, NUM_IDS));
    
    std::cout << "Testando hash v1 (MurmurHash-like)..." << std::endl;
    results.push_back(testHashFunction("v1 - MurmurHash", prune_hash_v1, probabilities, NUM_IDS));
    
    std::cout << "Testando hash v2 (com prob no seed)..." << std::endl;
    results.push_back(testHashFunction("v2 - Prob Seed", prune_hash_v2, probabilities, NUM_IDS));
    
    std::cout << "Testando hash v3 (ID+índice)..." << std::endl;
    results.push_back(testHashFunction("v3 - ID+Index", prune_hash_v3, probabilities, NUM_IDS));
    
    std::cout << "Testando hash v4 (xxHash-like)..." << std::endl;
    results.push_back(testHashFunction("v4 - xxHash", prune_hash_v4, probabilities, NUM_IDS));
    
    // Resultados comparativos
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "COMPARAÇÃO DE IMPLEMENTAÇÕES" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::left << std::setw(20) << "Implementação" 
              << std::setw(15) << "Padrões Únicos" 
              << std::setw(15) << "Taxa Unicidade" 
              << std::setw(15) << "Maior Colisão" 
              << std::setw(10) << "Meta?" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    HashTestResult* best = nullptr;
    
    for (auto& result : results) {
        bool meets_target = result.uniqueness_rate >= TARGET_UNIQUENESS;
        
        std::cout << std::left << std::setw(20) << result.name
                  << std::setw(15) << result.unique_patterns
                  << std::setw(15) << std::fixed << std::setprecision(4) << (result.uniqueness_rate * 100) << "%"
                  << std::setw(15) << result.max_collision
                  << std::setw(10) << (meets_target ? "✓ SIM" : "✗ NÃO") << std::endl;
        
        if (!best || result.uniqueness_rate > best->uniqueness_rate) {
            best = &result;
        }
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "RECOMENDAÇÃO" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    if (best && best->uniqueness_rate >= TARGET_UNIQUENESS) {
        std::cout << "✓ SUCESSO! A implementação '" << best->name 
                  << "' atende à meta de " << (TARGET_UNIQUENESS * 100) << "%" << std::endl;
        std::cout << "Taxa de unicidade: " << std::fixed << std::setprecision(2) 
                  << (best->uniqueness_rate * 100) << "%" << std::endl;
        std::cout << "Padrões únicos: " << best->unique_patterns << "/" << NUM_IDS << std::endl;
    } else if (best) {
        std::cout << "⚠ Melhor opção: '" << best->name << "' com " 
                  << std::fixed << std::setprecision(2) << (best->uniqueness_rate * 100) 
                  << "% de unicidade" << std::endl;
        std::cout << "Ainda não atinge a meta de " << (TARGET_UNIQUENESS * 100) << "%" << std::endl;
        std::cout << "Considere usar uma abordagem diferente (ex: hash por posição)" << std::endl;
    }
    
    // Mostra código da melhor implementação
    if (best) {
        std::cout << "\nCÓDIGO DA MELHOR IMPLEMENTAÇÃO (" << best->name << "):" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        if (best->name == "v1 - MurmurHash") {
            std::cout << R"(int prune_hash(int id_val, float prob) {
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;
    if (id_val == 0) return 1;
    
    uint32_t hash_val = (uint32_t)id_val;
    hash_val ^= hash_val >> 16;
    hash_val *= 0x85ebca6b;
    hash_val ^= hash_val >> 13;
    hash_val *= 0xc2b2ae35;
    hash_val ^= hash_val >> 16;
    
    return ((hash_val & 0x7FFFFFFF) < (uint32_t)(prob * 2147483647.0f)) ? 1 : 0;
})" << std::endl;
        } else if (best->name == "v4 - xxHash") {
            std::cout << R"(int prune_hash(int id_val, float prob) {
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;
    if (id_val == 0) return 1;
    
    const uint32_t PRIME32_1 = 0x9E3779B1;
    const uint32_t PRIME32_2 = 0x85EBCA77;
    const uint32_t PRIME32_3 = 0xC2B2AE3D;
    const uint32_t PRIME32_4 = 0x27D4EB2F;
    const uint32_t PRIME32_5 = 0x165667B1;
    
    uint32_t h32 = PRIME32_5 + 4;
    h32 += (uint32_t)id_val * PRIME32_3;
    h32 = ((h32 << 17) | (h32 >> 15)) * PRIME32_4;
    
    h32 ^= h32 >> 15;
    h32 *= PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= PRIME32_3;
    h32 ^= h32 >> 16;
    
    return ((h32 & 0x7FFFFFFF) < (uint32_t)(prob * 2147483647.0f)) ? 1 : 0;
})" << std::endl;
        }
    }
    
    return 0;
}