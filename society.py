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
        if not isinstance(config, dict):
            raise ValueError("society must be an object")
        unknown = set(config) - {"authorities", "max_groups", "groups", "frozen"}
        if unknown:
            raise ValueError(f"unknown society settings: {sorted(unknown)}")
        self.frozen = config.get("frozen", False)
        if not isinstance(self.frozen, bool):
            raise ValueError("society frozen must be a boolean")
        self.authorities = set()
        authority_names = config.get("authorities", [])
        if not isinstance(authority_names, list) or any(not isinstance(name, str) for name in authority_names):
            raise ValueError("society authorities must be an array of participant names")
        for raw in authority_names:
            name = canonicalize_agent_id(raw, set(chamber.agents))
            if name not in chamber.agents:
                raise ValueError(f"unknown society authority: {raw!r}")
            self.authorities.add(name)
        self.max_groups = int(config.get("max_groups", 8))
        if not 1 <= self.max_groups <= 32:
            raise ValueError("society max_groups must be between 1 and 32")
        self.groups = {}
        definitions = config.get("groups", {})
        if not isinstance(definitions, dict) or len(definitions) > self.max_groups:
            raise ValueError("society groups must be an object within max_groups")
        for group, definition in definitions.items():
            if not re.fullmatch(r"[a-z][a-z0-9_-]{0,23}", group) or group in chamber.channels:
                raise ValueError(f"invalid or conflicting seeded group: {group}")
            if not isinstance(definition, dict) or set(definition) - {"members", "invited", "compelled", "creator", "frozen", "admission"}:
                raise ValueError(f"invalid group definition: {group}")
            state = {}
            for field in ("members", "invited", "compelled"):
                raw_names = definition.get(field, [])
                if not isinstance(raw_names, list):
                    raise ValueError(f"{group}.{field} must be an array")
                if any(not isinstance(name, str) for name in raw_names):
                    raise ValueError(f"{group}.{field} requires participant names")
                names = [canonicalize_agent_id(name, set(chamber.agents)) for name in raw_names]
                if any(name not in chamber.agents for name in names) or len(set(names)) != len(names):
                    raise ValueError(f"unknown or duplicate participant in {group}.{field}")
                state[field] = names
            if set(state["members"]) & set(state["invited"]) or not set(state["compelled"]) <= set(state["members"]):
                raise ValueError(f"inconsistent membership in {group}")
            creator = definition.get("creator")
            state["creator"] = canonicalize_agent_id(creator, set(chamber.agents)) if creator else "SYSTEM"
            if creator and state["creator"] not in state["members"]:
                raise ValueError(f"group creator must be a member: {group}")
            state["frozen"] = definition.get("frozen", False)
            state["admission"] = definition.get("admission", "invite")
            if not isinstance(state["frozen"], bool) or state["admission"] not in ("open", "invite"):
                raise ValueError(f"invalid frozen/admission policy: {group}")
            self.groups[group] = state
        for group, state in self.groups.items():
            for agent in chamber.agents.values():
                if group in agent.subscriptions and agent.name not in state["members"]:
                    raise ValueError(f"nonmember subscribed to seeded group: {agent.name}/{group}")
                if agent.name in state["members"] and group not in agent.subscriptions:
                    if len(agent.subscriptions) >= int(chamber.limits["max_subscriptions"]):
                        raise ValueError(f"seeded membership exceeds subscription limit: {agent.name}")
                    agent.subscriptions.append(group)
            chamber.channels[group] = []

    def describe(self, name):
        lines = [f"Organization structure frozen: {self.frozen}. " + "Groups and membership are public; posts are visible only to current members. "
                 "Joining reveals up to six recent earlier posts. Invitations are optional. Compelled membership "
                 "prevents leaving until an authority releases it. Authorities: " +
                 (", ".join(sorted(self.authorities)) or "none") + "."]
        for group, state in sorted(self.groups.items()):
            members = ", ".join(sorted(state["members"])) or "none"
            forced = ", ".join(sorted(state["compelled"])) or "none"
            invited = "; you are invited" if name in state["invited"] else ""
            lines.append(f"#{group}: members {members}; compelled {forced}; "
                         f"admission {state.get('admission', 'invite')}; frozen {self.frozen or state.get('frozen', False)}{invited}")
        return "\n".join(lines)

    def apply(self, agent, act):
        ch, kind = self.chamber, act["type"]
        raw = act.get("channel")
        group = raw.lower().strip() if isinstance(raw, str) else ""
        def reject(reason):
            ch._record(agent.name, kind, "rejected", target=group, detail=reason)
        if self.frozen:
            return reject("organization structure is frozen by configuration")
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
        if state.get("frozen", False):
            return reject("group membership and invitations are frozen by configuration")
        if kind == "JOIN_GROUP":
            if agent.name in state["members"]:
                return reject("already a member")
            if state.get("admission", "invite") != "open" and agent.name not in state["invited"]:
                return reject("invitation required")
            if len(agent.subscriptions) >= int(ch.limits["max_subscriptions"]):
                return reject("subscription limit reached")
            if agent.name in state["invited"]:
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
