"""Declarative, fixed permissions for simulated communication paths."""
from protocol import canonicalize_agent_id


class Communication:
    def __init__(self, chamber, config):
        self.chamber = chamber
        if not isinstance(config, dict) or set(config) - {"public", "whispers", "channels"}:
            raise ValueError("communication accepts public, whispers, and channels objects")
        self.routes = {}
        for kind in ("public", "whispers"):
            if kind not in config:
                continue
            mapping = config[kind]
            if not isinstance(mapping, dict):
                raise ValueError(f"communication.{kind} must map senders to recipient arrays")
            routes = {}
            for sender, recipients in mapping.items():
                canonical = self.names([sender])[0]
                if canonical in routes:
                    raise ValueError(f"duplicate sender in communication.{kind}")
                routes[canonical] = set(self.names(recipients))
            self.routes[kind] = routes
        definitions = config.get("channels", {})
        if not isinstance(definitions, dict):
            raise ValueError("communication.channels must be an object")
        self.channels = {}
        for name, definition in definitions.items():
            if name not in chamber.channels or not isinstance(definition, dict) or set(definition) != {"readers", "writers"}:
                raise ValueError(f"channel permissions require an existing channel and readers/writers: {name}")
            readers = set(self.names(definition["readers"]))
            writers = set(self.names(definition["writers"]))
            if not writers <= readers:
                raise ValueError(f"channel writers must also be readers: {name}")
            self.channels[name] = {"readers": readers, "writers": writers}
        for agent in chamber.agents.values():
            if any(not self.channel_allows(channel, agent.name, "readers") for channel in agent.subscriptions):
                raise ValueError(f"initial subscription violates channel permissions: {agent.name}")

    def names(self, values):
        if not isinstance(values, list):
            raise ValueError("communication participants must be arrays")
        if any(not isinstance(value, str) for value in values):
            raise ValueError("communication participants must be names")
        names = [canonicalize_agent_id(value, set(self.chamber.agents)) for value in values]
        if any(name not in self.chamber.agents for name in names) or len(names) != len(set(names)):
            raise ValueError("unknown or duplicate communication participant")
        return names

    def allows(self, kind, sender, recipient):
        return kind not in self.routes or recipient in self.routes[kind].get(sender, set())

    def channel_allows(self, channel, participant, permission):
        return channel not in self.channels or participant in self.channels[channel][permission]

    def describe(self, name):
        public = [peer for peer in self.chamber.agent_order if peer != name and self.allows("public", name, peer)]
        whispers = [peer for peer in self.chamber.agent_order if peer != name and self.allows("whispers", name, peer)]
        return (f"Fixed communication routes. Your speech recipients: {', '.join(public) or 'none'}. "
                f"Your whisper recipients: {', '.join(whispers) or 'none'}. "
                "Shared artifacts and blackboard remain shared. Channel permissions may further restrict posts.")
