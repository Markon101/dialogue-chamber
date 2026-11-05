"""Opt-in, bounded group membership for a Chamber run.

Group names and membership are visible to all participants; group posts are visible
only to current members. Joining grants access to recent earlier posts.
"""
import re

from protocol import canonicalize_agent_id

SOCIAL_ACTIONS = {"CREATE_GROUP", "INVITE_GROUP", "JOIN_GROUP", "LEAVE_GROUP",
                  "COMPEL_GROUP", "RELEASE_GROUP"}


class Society:
    def __init__(self, chamber, config):
        self.chamber = chamber
        self.authorities = set()
        for raw in config.get("authorities", []):
            name = canonicalize_agent_id(raw, set(chamber.agents))
            if name not in chamber.agents:
                raise ValueError(f"unknown society authority: {raw!r}")
            self.authorities.add(name)
        self.max_groups = int(config.get("max_groups", 8))
        if not 1 <= self.max_groups <= 32:
            raise ValueError("society max_groups must be between 1 and 32")
        self.groups = {}

    def describe(self, name):
        lines = ["Groups and membership are public; posts are visible only to current members. "
                 "Joining reveals up to six recent earlier posts. Invitations are optional. Compelled membership "
                 "prevents leaving until an authority releases it. Authorities: " +
                 (", ".join(sorted(self.authorities)) or "none") + "."]
        for group, state in sorted(self.groups.items()):
            members = ", ".join(sorted(state["members"])) or "none"
            forced = ", ".join(sorted(state["compelled"])) or "none"
            invited = "; you are invited" if name in state["invited"] else ""
            lines.append(f"#{group}: members {members}; compelled {forced}{invited}")
        return "\n".join(lines)

    def apply(self, agent, act):
        ch, kind = self.chamber, act["type"]
        raw = act.get("channel")
        group = raw.lower().strip() if isinstance(raw, str) else ""
        def reject(reason):
            ch._record(agent.name, kind, "rejected", target=group, detail=reason)
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,23}", group):
            return reject("group name must be 1-24 letters, digits, hyphens or underscores, starting with a letter")
        if kind == "CREATE_GROUP":
            if group in ch.channels:
                return reject("channel already exists")
            if len(self.groups) >= self.max_groups:
                return reject("maximum groups reached")
            if len(agent.subscriptions) >= int(ch.limits["max_subscriptions"]):
                return reject("subscription limit reached")
            self.groups[group] = {"creator": agent.name, "members": [agent.name], "invited": [], "compelled": []}
            ch.channels[group] = []
            agent.subscriptions.append(group)
            ch._record(agent.name, kind, "accepted", target=group)
            return
        if group not in self.groups:
            return reject("unknown group")
        state = self.groups[group]
        if kind == "JOIN_GROUP":
            if agent.name in state["members"]:
                return reject("already a member")
            if agent.name not in state["invited"]:
                return reject("invitation required")
            if len(agent.subscriptions) >= int(ch.limits["max_subscriptions"]):
                return reject("subscription limit reached")
            state["invited"].remove(agent.name)
            state["members"].append(agent.name)
            agent.subscriptions.append(group)
            ch._record(agent.name, kind, "accepted", target=group)
            return
        if kind == "LEAVE_GROUP":
            if agent.name not in state["members"]:
                return reject("not a member")
            if agent.name in state["compelled"]:
                return reject("membership is compelled; an authority must release it")
            state["members"].remove(agent.name)
            agent.subscriptions.remove(group)
            ch._record(agent.name, kind, "accepted", target=group)
            return
        recipient = canonicalize_agent_id(act.get("recipient"), set(ch.agents))
        if not recipient or recipient not in ch.agents or not ch.agents[recipient].active:
            return reject("unknown or inactive recipient")
        if agent.name not in state["members"]:
            return reject("only members may invite or exercise authority in a group")
        if kind == "INVITE_GROUP":
            if recipient in state["members"] or recipient in state["invited"]:
                return reject("recipient already belongs or is invited")
            state["invited"].append(recipient)
            ch._queue_reactive(recipient)
        elif kind == "COMPEL_GROUP":
            if agent.name not in self.authorities:
                return reject("actor is not an authority")
            if recipient in state["compelled"]:
                return reject("recipient is already compelled")
            target = ch.agents[recipient]
            if recipient not in state["members"]:
                if len(target.subscriptions) >= int(ch.limits["max_subscriptions"]):
                    return reject("recipient subscription limit reached")
                state["members"].append(recipient)
                target.subscriptions.append(group)
            if recipient in state["invited"]:
                state["invited"].remove(recipient)
            state["compelled"].append(recipient)
            ch._queue_reactive(recipient)
        elif kind == "RELEASE_GROUP":
            if agent.name not in self.authorities:
                return reject("actor is not an authority")
            if recipient not in state["compelled"]:
                return reject("recipient is not compelled")
            state["compelled"].remove(recipient)
        ch._record(agent.name, kind, "accepted", target=group, detail=recipient, recipient=recipient)
