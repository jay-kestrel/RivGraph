      *> COBOL walker kernel for RivGraph's skeleton walk (spec art_QEojt4gd, D2).
      *>
      *> Numeric traversal only: extends one link path across the skeleton
      *> until the Python walker must make a decision. Every decision that
      *> depends on CPython set iteration order (branchpoint clustering,
      *> emanator ordering, first-step selection) stays in Python, which hands
      *> the kernel canonicalized inputs:
      *>
      *>   LS-SKEL    padded binary skeleton, uint8, values 0/1
      *>   LS-BPMAP   per-pixel branchpoint verdicts: 0 no, 1 yes, 255 unknown
      *>   LS-CANT    sorted flat indices excluded while the link is short
      *>
      *> The kernel receives inputs, never draws randomness, and never
      *> allocates. Stop codes (LS-CODE):
      *>   0 parameter error (e.g. empty path)
      *>   1 endpoint reached (raw candidates empty)
      *>   2 emanating link (candidates empty after the short-link filter)
      *>   3 branchpoint reached by single-candidate step; path extended
      *>   4 junction: LS-NPOSS sorted candidates returned; path unchanged
      *>   5 single candidate lacks a branchpoint verdict (LS-POSS(1) set)
      *>   6 path buffer would overflow LS-CAP
       IDENTIFICATION DIVISION.
       PROGRAM-ID. WALKMARCH.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-END PIC S9(18) USAGE COMP-5.
       01 WS-ENDROW PIC S9(18) USAGE COMP-5.
       01 WS-ENDCOL PIC S9(18) USAGE COMP-5.
       01 WS-R PIC S9(18) USAGE COMP-5.
       01 WS-C PIC S9(18) USAGE COMP-5.
       01 WS-IDX PIC S9(18) USAGE COMP-5.
       01 WS-NEXT PIC S9(18) USAGE COMP-5.
       01 WS-DR USAGE BINARY-LONG.
       01 WS-DC USAGE BINARY-LONG.
       01 WS-I USAGE BINARY-LONG.
       01 WS-J USAGE BINARY-LONG.
       01 WS-KEEP USAGE BINARY-LONG.
       01 WS-BLOCKED USAGE BINARY-LONG.
       01 WS-SWAPPED USAGE BINARY-LONG.
       01 WS-N USAGE BINARY-LONG.
       01 WS-EXCL-A PIC S9(18) USAGE COMP-5.
       01 WS-EXCL-B PIC S9(18) USAGE COMP-5.
       LINKAGE SECTION.
       01 LS-SKELBUF.
          05 LS-SKEL PIC X OCCURS 268435456.
       01 LS-BPMAPBUF.
          05 LS-BPMAP PIC X OCCURS 268435456.
       01 LS-ROWS USAGE BINARY-LONG.
       01 LS-COLS USAGE BINARY-LONG.
       01 LS-PATHBUF.
          05 LS-PATH PIC S9(18) USAGE COMP-5 OCCURS 268435455.
       01 LS-PATHLEN USAGE BINARY-LONG.
       01 LS-CANTBUF.
          05 LS-CANT PIC S9(18) USAGE COMP-5 OCCURS 268435455.
       01 LS-NCNT USAGE BINARY-LONG.
       01 LS-CAP PIC S9(18) USAGE COMP-5.
       01 LS-NPOSS USAGE BINARY-LONG.
       01 LS-POSSBUF.
          05 LS-POSS PIC S9(18) USAGE COMP-5 OCCURS 8.
       01 LS-CODE USAGE BINARY-LONG.
       PROCEDURE DIVISION USING LS-SKELBUF LS-BPMAPBUF LS-ROWS LS-COLS
                                LS-PATHBUF LS-PATHLEN LS-CANTBUF LS-NCNT
                                LS-CAP LS-NPOSS LS-POSSBUF LS-CODE.
           MOVE 0 TO LS-CODE
           IF LS-PATHLEN < 1
               GOBACK
           END-IF
           MOVE 1 TO LS-NPOSS
           PERFORM UNTIL LS-CODE > 0
      *> Candidates: 8-connected on-pixels around the path end, minus the
      *> last two path pixels (walk.walkable_neighbors semantics).
               COMPUTE WS-END = LS-PATH(LS-PATHLEN)
               COMPUTE WS-ENDROW = WS-END / LS-COLS
               COMPUTE WS-ENDCOL = WS-END - WS-ENDROW * LS-COLS
               IF LS-PATHLEN >= 2
                   COMPUTE WS-EXCL-A = LS-PATH(LS-PATHLEN - 1)
               ELSE
                   MOVE -1 TO WS-EXCL-A
               END-IF
               MOVE WS-END TO WS-EXCL-B
               MOVE 0 TO WS-N
               PERFORM VARYING WS-I FROM 1 BY 1 UNTIL WS-I > 8
                   EVALUATE WS-I
                       WHEN 1
                           MOVE -1 TO WS-DR
                           MOVE -1 TO WS-DC
                       WHEN 2
                           MOVE -1 TO WS-DR
                           MOVE 0 TO WS-DC
                       WHEN 3
                           MOVE -1 TO WS-DR
                           MOVE 1 TO WS-DC
                       WHEN 4
                           MOVE 0 TO WS-DR
                           MOVE -1 TO WS-DC
                       WHEN 5
                           MOVE 0 TO WS-DR
                           MOVE 1 TO WS-DC
                       WHEN 6
                           MOVE 1 TO WS-DR
                           MOVE -1 TO WS-DC
                       WHEN 7
                           MOVE 1 TO WS-DR
                           MOVE 0 TO WS-DC
                       WHEN 8
                           MOVE 1 TO WS-DR
                           MOVE 1 TO WS-DC
                   END-EVALUATE
                   COMPUTE WS-R = WS-ENDROW + WS-DR
                   COMPUTE WS-C = WS-ENDCOL + WS-DC
                   IF WS-R >= 0 AND WS-R < LS-ROWS
                       AND WS-C >= 0 AND WS-C < LS-COLS
                       COMPUTE WS-IDX = WS-R * LS-COLS + WS-C
                       IF LS-SKEL(WS-IDX + 1) = X"01"
                           AND WS-IDX NOT = WS-EXCL-A
                           AND WS-IDX NOT = WS-EXCL-B
                           ADD 1 TO WS-N
                           MOVE WS-IDX TO LS-POSS(WS-N)
                       END-IF
                   END-IF
               END-PERFORM
               MOVE 0 TO WS-KEEP
      *> Endpoint check precedes the short-link filter, per the reference.
               IF WS-N = 0
                   MOVE 1 TO LS-CODE
               END-IF
               IF LS-CODE = 0 AND LS-PATHLEN < 4
                   PERFORM VARYING WS-I FROM 1 BY 1 UNTIL WS-I > WS-N
                       MOVE 0 TO WS-BLOCKED
                       PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > LS-NCNT
                           IF LS-CANT(WS-J) = LS-POSS(WS-I)
                               MOVE 1 TO WS-BLOCKED
                           END-IF
                       END-PERFORM
                       IF WS-BLOCKED = 0
                           ADD 1 TO WS-KEEP
                           MOVE LS-POSS(WS-I) TO LS-POSS(WS-KEEP)
                       END-IF
                   END-PERFORM
                   COMPUTE WS-N = WS-KEEP
               END-IF
               IF LS-CODE = 0 AND WS-N = 0
                   MOVE 2 TO LS-CODE
               END-IF
               IF LS-CODE = 0 AND WS-N > 1
                   PERFORM SORT-CANDIDATES
                   MOVE WS-N TO LS-NPOSS
                   MOVE 4 TO LS-CODE
               END-IF
               IF LS-CODE = 0 AND WS-N = 1
                   MOVE LS-POSS(1) TO WS-NEXT
                   EVALUATE LS-BPMAP(WS-NEXT + 1)
                       WHEN X"FF"
                           MOVE 5 TO LS-CODE
                       WHEN X"01"
                           ADD 1 TO LS-PATHLEN
                           IF LS-PATHLEN > LS-CAP
                               COMPUTE LS-PATHLEN = LS-PATHLEN - 1
                               MOVE 6 TO LS-CODE
                           ELSE
                               MOVE WS-NEXT TO LS-PATH(LS-PATHLEN)
                               MOVE 3 TO LS-CODE
                           END-IF
                       WHEN OTHER
                           ADD 1 TO LS-PATHLEN
                           IF LS-PATHLEN > LS-CAP
                               COMPUTE LS-PATHLEN = LS-PATHLEN - 1
                               MOVE 6 TO LS-CODE
                           ELSE
                               MOVE WS-NEXT TO LS-PATH(LS-PATHLEN)
      *> code stays 0: keep marching
                           END-IF
                   END-EVALUATE
               END-IF
           END-PERFORM
           GOBACK.
       SORT-CANDIDATES.
      *> Ascending insertion of the junction candidates (canonical order).
           MOVE 1 TO WS-SWAPPED
           PERFORM UNTIL WS-SWAPPED = 0
               MOVE 0 TO WS-SWAPPED
               PERFORM VARYING WS-I FROM 1 BY 1 UNTIL WS-I >= WS-N
                   IF LS-POSS(WS-I) > LS-POSS(WS-I + 1)
                       COMPUTE WS-IDX = LS-POSS(WS-I)
                       MOVE LS-POSS(WS-I + 1) TO LS-POSS(WS-I)
                       MOVE WS-IDX TO LS-POSS(WS-I + 1)
                       MOVE 1 TO WS-SWAPPED
                   END-IF
               END-PERFORM
           END-PERFORM.
