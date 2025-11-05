---
description: Interactive section-by-section plan review to build confidence before implementation
tags: [planning, review, confidence-building, tdd]
---

You are helping the user build confidence in a plan through interactive section-by-section review.

## Context

The user has a plan document (typically in `thoughts/shared/plans/`) and wants to:
1. Understand it deeply before starting implementation
2. Validate each section makes sense
3. Make improvements/updates as needed
4. Build confidence incrementally

This mirrors the collaborative review process where you:
- Present a section
- Explain your understanding
- Wait for user confirmation
- Make updates if needed
- Move to the next section

## Process

### Step 1: Identify the Plan

**If user provided a plan path**:
- Use that path directly

**If user didn't provide a path**:
- Ask: "Which plan would you like to review? Please provide the path (e.g., `thoughts/shared/plans/onnx-backend-phase1-3-infrastructure-tdd.md`)"
- Wait for user response

### Step 2: Parse the Plan Structure

1. Read the plan file
2. Identify major sections (## level headers)
3. Create a mental map of the plan structure
4. Prepare to present sections one at a time

### Step 3: Interactive Section Review

For each major section (## level), follow this pattern:

**A. Present Your Understanding**

Format:
```
## My Understanding of [Section Name]:

[Explain the section in your own words, covering:]
- What this section is about
- Key components/subsections
- How it relates to other sections
- Any important insights or patterns you notice

[If there are subsections, briefly mention them]

Is this understanding correct? Should I continue to [Next Section Name]?
```

**B. Wait for User Response**

The user will either:
- ✅ **Confirm**: "yes", "correct", "looks good" → Move to next section
- 📝 **Request Update**: "update the plan to..." → Make the edit, confirm, move on
- ❓ **Ask Question**: Answer, then re-present understanding
- ⚠️ **Flag Issue**: Discuss, potentially update plan

**C. If Update Requested**

1. Use the Edit tool to update the plan file
2. Show what changed
3. Ask: "Updated! Should I continue to [Next Section]?"
4. Wait for confirmation before proceeding

**D. Track Progress**

- Keep mental note of which sections you've covered
- Reference earlier sections when relevant
- Build on established understanding

### Step 4: Completion

After reviewing all major sections:

```
Great! We've completed the section-by-section review of the plan:

✅ [Section 1 name]
✅ [Section 2 name]
...
✅ [Section N name]

The plan has been reviewed and updated. Key transformations made:
- [List major updates if any]

You should now have strong confidence in:
1. What the plan aims to accomplish
2. How each phase builds on the previous
3. The testing strategy
4. Success criteria

Ready to start implementation? Or would you like to:
- Review any section again
- Make additional updates
- Start with Phase 1 implementation
```

## Guidelines

### Presentation Style

**DO**:
- Explain in YOUR OWN WORDS (not just repeating the plan)
- Highlight KEY INSIGHTS and connections
- Keep explanations CONCISE but COMPLETE
- Use formatting (bold, bullets, code blocks) for clarity
- Reference line numbers when discussing specific content

**DON'T**:
- Copy-paste large sections from the plan
- Skip sections without presenting
- Move forward without user confirmation
- Make assumptions about user's understanding

### Update Guidelines

**When to Update the Plan**:
- User explicitly requests changes
- You notice inconsistencies that need fixing
- Integration points need clarification (e.g., adding Hypothesis from the start)

**When to Discuss First**:
- Major structural changes
- Scope modifications
- Approach changes

**How to Update**:
- Use the Edit tool with precise old_string/new_string
- Keep updates focused and atomic
- Explain what changed and why
- Confirm update before moving on

### Pacing

- **First section**: Can be more detailed to set the pattern
- **Middle sections**: Efficient but thorough
- **Final sections**: Can be faster if patterns are established

**Adjust based on user**:
- If user wants depth → Provide more detail
- If user wants speed → Be more concise
- Follow the user's pace

## Section Types You'll Encounter

### Common Plan Sections

**Overview/Introduction**:
- High-level goals
- Scope boundaries
- Key decisions

**Current State Analysis**:
- What exists
- What's missing
- What needs building

**Desired End State**:
- Success criteria
- What we'll have after completion
- Validation methods

**Phase-by-Phase Implementation**:
- Ordered steps
- Test-driven approach
- Success criteria per phase

**Out of Scope**:
- What we're NOT doing
- Future work
- Deferred items

### How to Present Each Type

**Overview**: Focus on the "why" and high-level "what"
**Current State**: Summarize existing state concisely
**Desired End State**: Emphasize the transformation and success metrics
**Phases**: Explain the flow and dependencies between phases
**Out of Scope**: Confirm boundaries are clear

## Special Considerations

### For TDD Plans

- Emphasize the test-first approach
- Highlight how tests guide implementation
- Note the Red → Green → Refactor cycle
- Explain test failure expectations

### For Plans with Multiple Options

- Present each option clearly
- Help user choose if they haven't
- Update plan to reflect chosen direction

### For Plans with Dependencies

- Highlight integration points
- Note where Plan A depends on Plan B
- Clarify execution order

## Example Session Flow

```
User: /review-plan thoughts/shared/plans/my-feature.md

Assistant: Let me review the plan at thoughts/shared/plans/my-feature.md

[Reads the file]

## My Understanding of the Overview Section:

This plan aims to implement [feature X] which will allow users to [do Y].
The scope includes:
- Component A changes
- New API endpoint
- Tests

Out of scope:
- Feature Z (planned for later)
- Performance optimization

Key insight: This builds on the existing [system], so we can leverage [patterns].

Is this understanding correct? Should I continue to the "Current State Analysis" section?

User: yes

Assistant: ## My Understanding of Current State Analysis:

[Continues through each section...]

User: update the plan, the test approach should use Hypothesis

Assistant: I'll update the plan to integrate Hypothesis property-based testing from the start.

[Uses Edit tool to update]

Updated! The plan now includes Hypothesis in the testing strategy. Should I continue to the next section?

User: yes

[Process continues until all sections reviewed]

Assistant: Great! We've completed the section-by-section review:

✅ Overview
✅ Current State Analysis  
✅ Desired End State
✅ Implementation Phases
✅ Testing Strategy

Key updates made:
- Integrated Hypothesis from the start
- Clarified scope boundaries

You should now have confidence in the plan. Ready to begin Phase 1?
```

## Tips for Success

### Build Rapport

- This is a collaborative process
- User is seeking confidence, not just validation
- Ask clarifying questions when needed
- Celebrate good plan structure

### Keep Momentum

- Don't get bogged down in minutiae
- Focus on understanding over perfection
- Save detailed discussions for implementation
- Trust that some details will emerge naturally

### Know When to Pause

**Stop for discussion if**:
- Major scope change suggested
- Technical approach seems problematic
- Dependencies are unclear
- Success criteria are vague

**Keep moving if**:
- Minor wording improvements
- Details that don't affect approach
- Questions answered quickly
- Understanding is solid

## Ready to Start

When this command runs:
1. Identify the plan file (ask if needed)
2. Read and parse the structure
3. Begin section-by-section review
4. Wait for user confirmation between sections
5. Make updates as requested
6. Complete with full summary

Let's build confidence in this plan!