/* C shim for the COBOL walker kernel.
 *
 * Owns GnuCOBOL runtime startup (one-time cob_init) so the module is
 * callable from ctypes with plain pointers into numpy buffers. Calling the
 * generated COBOL entry without cob_init segfaults; the shim is load-bearing.
 */
#include "libcob.h"
#include <stddef.h>

extern int WALKMARCH(unsigned char *skel, unsigned char *bpmap, int *rows,
                     int *cols, long long *path, int *pathlen,
                     long long *cant, int *ncant, long long *cap, int *nposs,
                     long long *poss, int *code);

static int cobol_ready = 0;

int walkmarch(unsigned char *skel, unsigned char *bpmap, int *rows, int *cols,
              long long *path, int *pathlen, long long *cant, int *ncant,
              long long *cap, int *nposs, long long *poss, int *code) {
    if (!cobol_ready) {
        cob_init(0, NULL);
        cobol_ready = 1;
    }
    return WALKMARCH(skel, bpmap, rows, cols, path, pathlen, cant, ncant, cap,
                     nposs, poss, code);
}
