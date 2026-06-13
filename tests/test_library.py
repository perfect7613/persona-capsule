import pytest

from persona_capsule.identity import Principal
from persona_capsule.library import AuthenticationRequiredError, CapsuleLibrary
from persona_capsule.repository import (
    CapsuleNotFoundError,
    CapsuleRecord,
    InMemoryCapsuleRepository,
)

ALICE = Principal(user_id="hf:alice", username="alice", source="test")
BOB = Principal(user_id="hf:bob", username="bob", source="test")


def test_private_library_requires_authentication() -> None:
    library = CapsuleLibrary(InMemoryCapsuleRepository())

    with pytest.raises(AuthenticationRequiredError):
        library.list_capsules(None)


def test_list_only_returns_the_owners_capsules() -> None:
    library = CapsuleLibrary(
        InMemoryCapsuleRepository(
            [
                CapsuleRecord("alice-1", ALICE.user_id, "Alice One"),
                CapsuleRecord("bob-1", BOB.user_id, "Bob One"),
            ]
        )
    )

    assert [record.capsule_id for record in library.list_capsules(ALICE)] == ["alice-1"]
    assert [record.capsule_id for record in library.list_capsules(BOB)] == ["bob-1"]


def test_one_user_cannot_read_or_mutate_another_users_capsule() -> None:
    library = CapsuleLibrary(
        InMemoryCapsuleRepository([CapsuleRecord("alice-1", ALICE.user_id, "Alice One")])
    )

    with pytest.raises(CapsuleNotFoundError):
        library.get_capsule(BOB, "alice-1")

    with pytest.raises(CapsuleNotFoundError):
        library.rename_capsule(BOB, "alice-1", "Taken over")

    with pytest.raises(CapsuleNotFoundError):
        library.save_capsule(
            BOB,
            CapsuleRecord("alice-1", ALICE.user_id, "Taken over"),
        )


def test_delete_is_owner_scoped_and_idempotent() -> None:
    library = CapsuleLibrary(
        InMemoryCapsuleRepository([CapsuleRecord("alice-1", ALICE.user_id, "Alice One")])
    )

    with pytest.raises(CapsuleNotFoundError):
        library.delete_capsule(BOB, "alice-1")

    assert library.delete_capsule(ALICE, "alice-1") is True
    assert library.delete_capsule(ALICE, "alice-1") is False
