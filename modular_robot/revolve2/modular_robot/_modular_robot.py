from __future__ import annotations
import uuid

from .body.base import Body
from .brain import Brain


class ModularRobot:
    """A module robot consisting of a body and brain."""

    _uuid: uuid.UUID

    body: Body
    brain: Brain

    def __init__(self, body: Body, brain: Brain):
        """
        Initialize the ModularRobot.

        :param body: The body of the modular robot.
        :param brain: The brain of the modular robot.
        """
        self._uuid = uuid.uuid1()
        self.body = body
        self.brain = brain

    @property
    def uuid(self) -> uuid.UUID:
        """
        Get the uuid, used for identification.

        :returns: The uuid.
        """
        return self._uuid

    def __copy__(self) -> ModularRobot:
        """
        Makes a copy of the ModularRobot.

        :returns: The copy of the ModularRobot.
        """
        return type(self)(body=self.body, brain=self.brain)
