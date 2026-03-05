# Goofspiel

## The Game of Pure Strategy

Goofspiel — also called GOPS (Game of Pure Strategy) — is a two-player card game with a remarkable property: after the initial shuffle, there is no randomness whatsoever. Every outcome is determined entirely by the players' decisions.

The game was introduced as a tool for studying strategic reasoning and has become a staple example in combinatorial game theory and the study of Nash equilibria.

## Rules

Separate a standard deck into its four suits. Each player receives one suit as their **hand** (say Spades and Clubs). A third suit (Hearts) forms the **prize pile**, which is shuffled and placed face down. The fourth suit is set aside.

Each round:

1. The top prize card is turned face up, revealing its value.
2. Both players simultaneously select a card from their hand and place it face down.
3. Both cards are revealed. The higher card wins the prize. On a tie, the prize is discarded.
4. Played cards are removed from the game.

After all rounds, the player with the higher total prize value wins.

## Why "Pure Strategy"?

The name is precise. Once the prize order is fixed, Goofspiel is a **perfect information** game with **simultaneous moves** — there is no hidden information (both players can see which cards have been played) and no chance element. The only thing that matters is strategy.

This makes Goofspiel an ideal setting for studying **game-theoretic** concepts: dominated strategies, best responses, and Nash equilibria.

## Dominated Strategies

A strategy is **dominated** if there exists another strategy that performs at least as well in all scenarios and strictly better in at least one. In Goofspiel, some intuitive strategies are dominated:

**Always playing your highest card** is dominated. If a low prize (say 1) appears, spending your 10 to win 1 point is wasteful — you would have been better off playing a low card.

More precisely, consider a prize of value \( p \) and suppose you play card \( c \). Your **net gain** from winning is \( p \), but your **opportunity cost** is losing card \( c \) for future rounds. A rational strategy must weigh the prize value against the cost of the card spent.

## The Matching Heuristic

A natural strategy is **matching**: play the card whose value is closest to the prize value. Intuitively, this ensures you never overspend on a cheap prize or underspend on an expensive one.

If both players use matching, ties occur frequently and prizes are discarded. The game becomes about the rounds where the prize order creates asymmetries.

## Nash Equilibrium

For the full \( n \)-card Goofspiel, computing a Nash equilibrium is non-trivial. The game tree grows exponentially: with \( n \) cards per player and \( n! \) possible prize orderings, the strategy space is vast.

For small \( n \), equilibria can be computed by backward induction. Consider 3-card Goofspiel (cards 1, 2, 3). The prize card is revealed, and each player chooses from their remaining hand.

**Key insight**: In the Nash equilibrium of 3-card Goofspiel, players must **randomize** — any pure (deterministic) strategy can be exploited. If I know you always play your highest card for the highest prize, I can counter-bid efficiently.

A mixed strategy Nash equilibrium specifies a probability distribution over cards for each possible game state. For the full 13-card game, the equilibrium involves complex conditional probability distributions that depend on the entire history of play.

## Connection to Auctions

Goofspiel has a deep connection to **all-pay auctions**. Each round is essentially a sealed-bid auction where:

- The **prize** is the face-up card.
- The **bids** are the cards played.
- Both players **pay** their bid regardless of outcome (the card is consumed).
- The highest bidder wins the prize.

The theory of all-pay auctions, developed by Baye, Kovenock, and de Vries (1996), provides tools for analyzing equilibrium bidding behavior. In an all-pay auction with budget constraints (your remaining hand), optimal bidding requires balancing current-round competitiveness against future resource preservation.

## Strategies in the Animation

The animation on this page shows two AI players competing at Goofspiel with strategies that vary between games:

- **Mirror**: Play the card closest in value to the prize. A balanced strategy that avoids overspending.
- **Greedy**: Play your highest card for high-value prizes, your lowest for low-value prizes. Intuitive but exploitable.
- **Thrifty**: Underbid slightly — play a card just below the prize value to conserve high cards for later rounds.
- **Random**: Choose uniformly at random from remaining cards. A surprisingly competitive baseline.

Watch how different strategy matchups play out. Mirror tends to draw against itself. Greedy wins early rounds but runs out of high cards. Thrifty conserves resources but sometimes loses prizes it should have won.

## Complexity

Despite its simple rules, Goofspiel is computationally challenging:

- The game has \( (n!)^2 \) possible play sequences for \( n \)-card Goofspiel.
- Computing the Nash equilibrium for 13-card Goofspiel remains an active area of research.
- The game has been used as a benchmark for AI game-playing algorithms, including Monte Carlo tree search and regret minimization.

Rhoads and Bartholdi (2012) showed that even restricted versions of Goofspiel are computationally hard, connecting the game to problems in computational complexity theory.

## References

- M. L. Rhoads, J. J. Bartholdi III, "Computational complexity of a card game," *International Journal of Game Theory*, 2012.
- M. R. Baye, D. Kovenock, C. G. de Vries, "The all-pay auction with complete information," *Economic Theory*, 1996.
- D. Ross, "Game Theory," *The Stanford Encyclopedia of Philosophy*, 2019.
- J. H. Conway, *On Numbers and Games*, Academic Press, 1976.
