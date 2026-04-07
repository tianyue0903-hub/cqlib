// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <stdio.h>
#include <stdlib.h>

#include "cqlib_c.h"

int main() {
    // 1. Create a parameterized circuit
    CircuitWrapper* circuit = circuit_new(2);
    if (circuit == NULL) {
        printf("Failed to create circuit.\n");
        return 1;
    }

    // 2. Build parametric circuit: RX(theta) - RY(phi) - CZ - H
    ParameterWrapper* theta = param_parse("theta");
    ParameterWrapper* phi = param_parse("phi");

    circuit_rx_param(circuit, 0, theta);
    circuit_ry_param(circuit, 1, phi);
    circuit_cz(circuit, 0, 1);
    circuit_h(circuit, 0);

    // 3. Dump to QCIS (with symbolic parameters)
    char* qcis = qcis_dumps(circuit);
    if (qcis != NULL) {
        printf("QCIS (symbolic):\n%s\n", qcis);
        cstring_free(qcis);
    }

    // 4. Assign parameters: theta = 0.5, phi = 1.57
    CircuitWrapper* circuit_assigned = circuit_assign_params(circuit, "theta:0.5,phi:1.57");
    if (circuit_assigned == NULL) {
        printf("Failed to assign parameters.\n");
        return 1;
    }

    // 5. Dump to QCIS (with numeric values)
    qcis = qcis_dumps(circuit_assigned);
    if (qcis != NULL) {
        printf("QCIS (assigned):\n%s\n", qcis);
        cstring_free(qcis);
    }

    // 6. Cleanup
    param_free(theta);
    param_free(phi);
    circuit_free(circuit);
    circuit_free(circuit_assigned);

    return 0;
}
