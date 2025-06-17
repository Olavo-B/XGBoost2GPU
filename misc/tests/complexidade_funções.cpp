#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdint>
#include <functional>

#include <algorithm>
#include <iostream>
#include <vector>

// Função hash original
int prune_hash_original(int id_val, float prob) {
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;
    if (id_val == 0) return 1;
    unsigned int hash_val = id_val * 1103515245U + 12345U;
    hash_val ^= hash_val >> 16;
    return ((hash_val & 0x7FFFFFFF) < (unsigned int)(prob * 2147483647.0f)) ? 1 : 0;
}

// v1 - MurmurHash-like
int prune_hash_v1(int id_val, float prob) {
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
}

// v2 - Prob no seed
int prune_hash_v2(int id_val, float prob) {
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;
    if (id_val == 0) return 1;
    
    uint32_t prob_bits = *(uint32_t*)&prob;
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

// v3 - ID+índice (versão thread-safe)
int prune_hash_v3(int id_val, float prob, int prob_index = 0) {
    if (id_val == 0) return 1;
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;
    
    uint64_t combined = ((uint64_t)id_val << 32) | (uint32_t)prob_index;
    uint32_t hash_val = (uint32_t)(combined ^ (combined >> 32));
    hash_val ^= hash_val >> 16;
    hash_val *= 0x85ebca6b;
    hash_val ^= hash_val >> 13;
    hash_val *= 0xc2b2ae35;
    hash_val ^= hash_val >> 16;
    
    return ((hash_val & 0x7FFFFFFF) < (uint32_t)(prob * 2147483647.0f)) ? 1 : 0;
}

// v4 - xxHash-like
int prune_hash_v4(int id_val, float prob) {
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

struct ComplexityAnalysis {
    std::string name;
    int operations_count;
    double avg_time_ns;
    double relative_performance;
    std::string complexity_class;
    std::string memory_usage;
    std::string instruction_breakdown;
};

// Contador de operações para análise estática
struct OpCounter {
    int arithmetic = 0;    // +, -, *, /
    int bitwise = 0;       // ^, &, |, <<, >>
    int comparisons = 0;   // <, >, ==, !=
    int memory = 0;        // carregamentos/armazenamentos
    int branches = 0;      // if, return
    int casts = 0;         // type conversions
    
    int total() const {
        return arithmetic + bitwise + comparisons + memory + branches + casts;
    }
    
    std::string breakdown() const {
        return "Arith:" + std::to_string(arithmetic) + 
               " Bit:" + std::to_string(bitwise) + 
               " Cmp:" + std::to_string(comparisons) + 
               " Mem:" + std::to_string(memory) + 
               " Branch:" + std::to_string(branches) +
               " Cast:" + std::to_string(casts);
    }
};

// Análise estática das operações
OpCounter analyzeOriginal() {
    OpCounter ops;
    ops.comparisons = 3;  // prob <= 0.0f, prob >= 1.0f, id_val == 0
    ops.branches = 3;     // 3 returns
    ops.arithmetic = 2;   // *, +
    ops.bitwise = 3;      // ^, >>, &
    ops.casts = 2;        // (unsigned int)
    ops.memory = 1;       // prob load
    return ops;
}

OpCounter analyzeV1() {
    OpCounter ops;
    ops.comparisons = 3;  // prob <= 0.0f, prob >= 1.0f, id_val == 0
    ops.branches = 3;     // 3 returns
    ops.arithmetic = 2;   // 2 multiplications
    ops.bitwise = 6;      // 3 XOR, 3 shifts, 1 AND
    ops.casts = 2;        // (uint32_t)
    ops.memory = 1;       // prob load
    return ops;
}

OpCounter analyzeV2() {
    OpCounter ops;
    ops.comparisons = 3;  // prob <= 0.0f, prob >= 1.0f, id_val == 0
    ops.branches = 3;     // 3 returns
    ops.arithmetic = 2;   // 2 multiplications
    ops.bitwise = 7;      // 4 XOR, 3 shifts, 1 AND
    ops.casts = 3;        // 2 (uint32_t) + 1 reinterpret_cast
    ops.memory = 2;       // prob load + prob_bits
    return ops;
}

OpCounter analyzeV3() {
    OpCounter ops;
    ops.comparisons = 3;  // prob <= 0.0f, prob >= 1.0f, id_val == 0
    ops.branches = 3;     // 3 returns
    ops.arithmetic = 2;   // 2 multiplications
    ops.bitwise = 8;      // 1 OR, 3 shifts, 3 XOR, 1 AND
    ops.casts = 4;        // multiple casts for 64-bit operations
    ops.memory = 2;       // prob load + prob_index
    return ops;
}

OpCounter analyzeV4() {
    OpCounter ops;
    ops.comparisons = 3;  // prob <= 0.0f, prob >= 1.0f, id_val == 0
    ops.branches = 3;     // 3 returns
    ops.arithmetic = 5;   // +, *, 3 more *
    ops.bitwise = 7;      // shifts, XOR operations, AND
    ops.casts = 2;        // (uint32_t)
    ops.memory = 6;       // 5 constants + prob load
    return ops;
}

double benchmarkFunction(std::function<int(int, float)> func, 
                        const std::vector<float>& probs, 
                        int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    volatile int result = 0; // Prevent optimization
    for (int iter = 0; iter < iterations; iter++) {
        for (int id = 1; id < 1000; id++) {
            for (float prob : probs) {
                result += func(id, prob);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    return duration.count() / double(iterations * 999 * probs.size());
}

int main() {
    const std::string filename = "prune.csv";
    std::vector<float> probabilities = readProbabilities(filename);
    
    if (probabilities.empty()) {
        std::cerr << "Erro ao ler probabilidades!" << std::endl;
        return 1;
    }
    
    std::cout << std::string(90, '=') << std::endl;
    std::cout << "ANÁLISE DE COMPLEXIDADE COMPUTACIONAL - FUNÇÕES HASH" << std::endl;
    std::cout << std::string(90, '=') << std::endl;
    
    // Análise estática
    std::vector<ComplexityAnalysis> analyses;
    
    // Original
    OpCounter orig_ops = analyzeOriginal();
    ComplexityAnalysis orig;
    orig.name = "Original";
    orig.operations_count = orig_ops.total();
    orig.complexity_class = "O(1) - Linear";
    orig.memory_usage = "~8 bytes stack";
    orig.instruction_breakdown = orig_ops.breakdown();
    analyses.push_back(orig);
    
    // v1
    OpCounter v1_ops = analyzeV1();
    ComplexityAnalysis v1;
    v1.name = "v1 - MurmurHash";
    v1.operations_count = v1_ops.total();
    v1.complexity_class = "O(1) - Constant";
    v1.memory_usage = "~8 bytes stack";
    v1.instruction_breakdown = v1_ops.breakdown();
    analyses.push_back(v1);
    
    // v2
    OpCounter v2_ops = analyzeV2();
    ComplexityAnalysis v2;
    v2.name = "v2 - Prob Seed";
    v2.operations_count = v2_ops.total();
    v2.complexity_class = "O(1) - Constant";
    v2.memory_usage = "~12 bytes stack";
    v2.instruction_breakdown = v2_ops.breakdown();
    analyses.push_back(v2);
    
    // v3
    OpCounter v3_ops = analyzeV3();
    ComplexityAnalysis v3;
    v3.name = "v3 - ID+Index";
    v3.operations_count = v3_ops.total();
    v3.complexity_class = "O(1) - Constant";
    v3.memory_usage = "~16 bytes stack";
    v3.instruction_breakdown = v3_ops.breakdown();
    analyses.push_back(v3);
    
    // v4
    OpCounter v4_ops = analyzeV4();
    ComplexityAnalysis v4;
    v4.name = "v4 - xxHash";
    v4.operations_count = v4_ops.total();
    v4.complexity_class = "O(1) - Constant";
    v4.memory_usage = "~40 bytes stack";
    v4.instruction_breakdown = v4_ops.breakdown();
    analyses.push_back(v4);
    
    std::cout << "\n1. ANÁLISE ESTÁTICA (Contagem de Operações)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left << std::setw(20) << "Função" 
              << std::setw(12) << "Total Ops" 
              << std::setw(15) << "Classe Compl." 
              << std::setw(15) << "Uso Memória" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (const auto& analysis : analyses) {
        std::cout << std::left << std::setw(20) << analysis.name
                  << std::setw(12) << analysis.operations_count
                  << std::setw(15) << analysis.complexity_class
                  << std::setw(15) << analysis.memory_usage << std::endl;
    }
    
    std::cout << "\n2. DETALHAMENTO DE OPERAÇÕES" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    for (const auto& analysis : analyses) {
        std::cout << analysis.name << ": " << analysis.instruction_breakdown << std::endl;
    }
    
    // Benchmark temporal
    std::cout << "\n3. BENCHMARK TEMPORAL (Execução Real)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Executando benchmarks com " << probabilities.size() << " probabilidades..." << std::endl;
    
    const int ITERATIONS = 100;
    
    // Benchmark todas as funções
    double orig_time = benchmarkFunction([](int id, float prob) { return prune_hash_original(id, prob); }, probabilities, ITERATIONS);
    double v1_time = benchmarkFunction([](int id, float prob) { return prune_hash_v1(id, prob); }, probabilities, ITERATIONS);
    double v2_time = benchmarkFunction([](int id, float prob) { return prune_hash_v2(id, prob); }, probabilities, ITERATIONS);
    double v4_time = benchmarkFunction([](int id, float prob) { return prune_hash_v4(id, prob); }, probabilities, ITERATIONS);
    
    // v3 precisa de tratamento especial
    auto start = std::chrono::high_resolution_clock::now();
    volatile int result = 0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int id = 1; id < 1000; id++) {
            for (size_t i = 0; i < probabilities.size(); i++) {
                result += prune_hash_v3(id, probabilities[i], i);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double v3_time = duration.count() / double(ITERATIONS * 999 * probabilities.size());
    
    // Atualiza análises com tempos
    analyses[0].avg_time_ns = orig_time;
    analyses[1].avg_time_ns = v1_time;
    analyses[2].avg_time_ns = v2_time;
    analyses[3].avg_time_ns = v3_time;
    analyses[4].avg_time_ns = v4_time;
    
    // Calcula performance relativa
    double baseline = orig_time;
    for (auto& analysis : analyses) {
        analysis.relative_performance = baseline / analysis.avg_time_ns;
    }
    
    std::cout << std::left << std::setw(20) << "Função" 
              << std::setw(15) << "Tempo (ns)" 
              << std::setw(15) << "Performance" 
              << std::setw(15) << "Eficiência" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (const auto& analysis : analyses) {
        std::string perf_indicator;
        if (analysis.relative_performance >= 1.1) perf_indicator = "🟢 Melhor";
        else if (analysis.relative_performance >= 0.9) perf_indicator = "🟡 Similar";
        else perf_indicator = "🔴 Pior";
        
        std::cout << std::left << std::setw(20) << analysis.name
                  << std::setw(15) << std::fixed << std::setprecision(2) << analysis.avg_time_ns
                  << std::setw(15) << std::fixed << std::setprecision(2) << analysis.relative_performance << "x"
                  << std::setw(15) << perf_indicator << std::endl;
    }
    
    std::cout << "\n4. ANÁLISE DETALHADA DE COMPLEXIDADE" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << "\n🔍 COMPLEXIDADE TEMPORAL:" << std::endl;
    std::cout << "• Todas as funções: O(1) - Tempo constante" << std::endl;
    std::cout << "• Diferenças estão no número de instruções por execução" << std::endl;
    std::cout << "• Original: " << orig_ops.total() << " ops | Mais rápida: " << v1_ops.total() << " ops" << std::endl;
    
    std::cout << "\n💾 COMPLEXIDADE ESPACIAL:" << std::endl;
    std::cout << "• Original: ~8 bytes (2 variáveis locais)" << std::endl;
    std::cout << "• v1: ~8 bytes (1 variável local)" << std::endl;
    std::cout << "• v2: ~12 bytes (3 variáveis locais)" << std::endl;
    std::cout << "• v3: ~16 bytes (operações 64-bit)" << std::endl;
    std::cout << "• v4: ~40 bytes (5 constantes + variáveis)" << std::endl;
    
    std::cout << "\n⚡ CONSIDERAÇÕES DE PERFORMANCE:" << std::endl;
    std::cout << "• Funções com menos multiplicações tendem a ser mais rápidas" << std::endl;
    std::cout << "• Operações bitwise são geralmente mais rápidas que aritméticas" << std::endl;
    std::cout << "• Constantes pré-definidas podem causar cache misses" << std::endl;
    std::cout << "• Reinterpret_cast pode ter overhead dependendo da arquitetura" << std::endl;
    
    // Recomendação final
    auto fastest = std::min_element(analyses.begin(), analyses.end(),
        [](const ComplexityAnalysis& a, const ComplexityAnalysis& b) {
            return a.avg_time_ns < b.avg_time_ns;
        });
    
    std::cout << "\n✅ RECOMENDAÇÃO:" << std::endl;
    std::cout << "Função mais rápida: " << fastest->name << std::endl;
    std::cout << "Considere balancear performance vs qualidade do hash!" << std::endl;
    
    return 0;
}   