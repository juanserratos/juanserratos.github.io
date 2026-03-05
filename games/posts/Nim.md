# Nim

## The Game

Nim is one of the oldest and most fundamental combinatorial games. Two players alternate turns. On each turn, a player must remove at least one object from a single pile. In **normal play**, the player who takes the last object wins; in **misère play**, the last player to take loses.

The game was first rigorously analyzed by Charles L. Bouton in 1901–1902, who gave it its name and discovered the complete winning strategy.

## Rules

Given \( n \) piles with sizes \( x_1, x_2, \ldots, x_n \):

1. Players alternate turns.
2. On your turn, choose one pile \( i \) and remove any number \( k \geq 1 \) of objects from it, leaving \( x_i - k \geq 0 \).
3. The player who takes the last object wins (normal play convention).

## The XOR Strategy

The key insight is the **nim-value** (or nim-sum) of a position:

\[
s = x_1 \oplus x_2 \oplus \cdots \oplus x_n
\]

where \( \oplus \) denotes the bitwise exclusive-or (XOR) operation.

**Bouton's Theorem.** A position is a losing position (for the player about to move) if and only if \( s = 0 \).

### Why does this work?

The proof has two parts:

**1. The terminal position has nim-sum 0.** When all piles are empty, \( 0 \oplus 0 \oplus \cdots \oplus 0 = 0 \). The player whose turn it is has no move — they've lost (their opponent took the last object).

**2. From any position with \( s \neq 0 \), there exists a move to a position with \( s = 0 \).** Let \( d \) be the position of the highest set bit in \( s \). There must be some pile \( x_i \) that has a 1 in bit position \( d \). Set \( x_i' = x_i \oplus s \). Then:
- \( x_i' < x_i \) (since bit \( d \) flips from 1 to 0, and all higher bits are unchanged).
- The new nim-sum is \( s \oplus x_i \oplus x_i' = s \oplus x_i \oplus (x_i \oplus s) = 0 \).

So we can remove \( x_i - x_i' \) objects from pile \( i \) to reach a position with nim-sum 0.

**3. From any position with \( s = 0 \), every move leads to a position with \( s \neq 0 \).** If we change \( x_i \) to \( x_i' < x_i \), the new nim-sum is \( s \oplus x_i \oplus x_i' = x_i \oplus x_i' \neq 0 \) (since \( x_i \neq x_i' \)).

Together, these facts mean the player facing \( s = 0 \) will always face \( s = 0 \) — and eventually face the terminal position (all zeros). The player facing \( s \neq 0 \) can always force \( s = 0 \) on their opponent.

## Example

Consider piles \( (3, 5, 7) \):

\[
3 \oplus 5 \oplus 7 = 011 \oplus 101 \oplus 111 = 001
\]

Since \( s = 1 \neq 0 \), the first player wins. One winning move: reduce pile 1 from 3 to 2 (take 1), giving \( (2, 5, 7) \):

\[
2 \oplus 5 \oplus 7 = 010 \oplus 101 \oplus 111 = 000
\]

Now the opponent faces nim-sum 0 and is in a losing position.

## The Sprague–Grundy Connection

Nim is not just another game — it is **the** fundamental impartial game. The **Sprague–Grundy theorem** states:

> Every impartial game under normal play convention is equivalent to a single Nim heap of some size \( g \), called the **Grundy value** (or nimber) of the position.

This means that any impartial game — no matter how complex its rules — can be analyzed by computing Grundy values. For a game that decomposes into independent subgames (like the piles in Nim), the overall Grundy value is the XOR of the individual Grundy values.

The Grundy value of a position \( P \) is defined recursively:

\[
\mathcal{G}(P) = \text{mex}\bigl(\{\mathcal{G}(Q) : Q \text{ is reachable from } P \text{ in one move}\}\bigr)
\]

where \( \text{mex}(S) \) is the minimum excludant — the smallest non-negative integer not in \( S \).

For a Nim pile of size \( n \), the reachable positions have sizes \( 0, 1, \ldots, n-1 \), so \( \mathcal{G}(n) = \text{mex}(\{0, 1, \ldots, n-1\}) = n \). The Grundy value of a Nim pile is just its size — which is why XOR of pile sizes gives the complete analysis.

## The Card Variant

The animation on this page shows a card-based variant: piles of playing cards replace the traditional stones or counters. The rules are identical — players remove cards from a single pile on each turn. Two AI players compete with strategies that vary between rounds:

- **Optimal**: plays the XOR strategy described above.
- **Greedy**: always takes from the largest pile.
- **Random**: picks a pile and quantity uniformly at random.

Watch how the optimal strategy dominates when paired against greedy or random play, but two optimal players reach an interesting equilibrium determined entirely by the starting position.

## References

- C. L. Bouton, "Nim, a game with a complete mathematical theory," *Annals of Mathematics*, 1901–1902.
- R. P. Sprague, "Über mathematische Kampfspiele," *Tôhoku Mathematical Journal*, 1935.
- P. M. Grundy, "Mathematics and games," *Eureka*, 1939.
- E. R. Berlekamp, J. H. Conway, R. K. Guy, *Winning Ways for Your Mathematical Plays*, 1982.
