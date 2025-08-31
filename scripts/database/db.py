import datetime
import logging
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from .db_classes import User, Message, UnknownMessage, UserMessageContext, UserRewardsRedeemed

# Disable verbose SQLAlchemy engine logs
logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)


class DatabaseManager:
    def __init__(self, db_url="sqlite:///scripts/database/twitch.db"):
        self.engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
        Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.session = Session()

    # --------------------
    # User Management
    # --------------------
    def record_user(self, twitch_username, twitch_user_id):
        q = self.session.query(User).filter(User.twitch_user_id == twitch_user_id)
        exists = self.session.query(q.exists()).scalar()

        if not exists:
            u = User(twitch_user_id, twitch_username)
            self.session.add(u)
            self.session.commit()
            print(f"User {twitch_username} has been added into the database")
            return

        if exists and q.first().twitch_username != twitch_username:
            past_username = q.first().twitch_username
            q.first().twitch_past_username = past_username
            q.first().twitch_username = twitch_username
            self.session.commit()
            print(f"User {twitch_username} has changed their name from {past_username}")

    # --------------------
    # Messages
    # --------------------
    def record_message_context(self, twitch_user_id, context):
        msg_con = UserMessageContext(twitch_user_id, context)
        self.session.add(msg_con)
        self.session.commit()

        # cleanup old contexts (older than 5 minutes)
        now = datetime.datetime.utcnow()
        cutoff_time = now - datetime.timedelta(minutes=5)
        old_contexts = (
            self.session.query(UserMessageContext)
            .filter(UserMessageContext.twitch_user_id == twitch_user_id)
            .filter(UserMessageContext.time_created < cutoff_time)
        )
        old_contexts.delete(synchronize_session=False)
        self.session.commit()

    def record_message(self, twitch_user_id, text, unknown=False):
        if unknown:
            unknown_message = UnknownMessage(twitch_user_id, text)
            self.session.add(unknown_message)
            self.session.commit()
            return

        q = (
            self.session.query(Message)
            .filter(Message.twitch_user_id == twitch_user_id)
            .first()
        )

        if q is None:
            message = Message(twitch_user_id, text)
            self.session.add(message)
        else:
            q.message_text = text

        self.session.commit()

    # --------------------
    # Rewards
    # --------------------
    def record_redeem(self, twitch_user_id, reward_id):
        msg_con = UserRewardsRedeemed(twitch_user_id, reward_id)
        self.session.add(msg_con)
        self.session.commit()

    # --------------------
    # Queries
    # --------------------
    def check_message_context(self, twitch_user_id):
        q = (
            self.session.query(UserMessageContext.message_context)
            .filter(UserMessageContext.twitch_user_id == twitch_user_id)
            .all()
        )
        return [r[0] for r in q]

    def check_reward_redeem_count(self, twitch_user_id, reward_id):
        q = (
            self.session.query(func.count(UserRewardsRedeemed.redeem_id))
            .filter(UserRewardsRedeemed.twitch_user_id == twitch_user_id)
            .filter(UserRewardsRedeemed.reward_id == reward_id)
        )
        return q.scalar()

    def check_best_headpatter(self):
        q = (
            self.session.query(
                UserRewardsRedeemed.twitch_user_id,
                func.count(UserRewardsRedeemed.twitch_user_id),
            )
            .filter(
                UserRewardsRedeemed.reward_id
                == "3d5f70df-61c1-4c47-bad9-4b5885ff3f76"
            )
            .group_by(UserRewardsRedeemed.twitch_user_id)
            .order_by(func.count(UserRewardsRedeemed.twitch_user_id).desc())
            .first()
        )

        if not q:
            return None

        username = (
            self.session.query(User.twitch_username)
            .filter(User.twitch_user_id == q[0])
            .first()[0]
        )
        return {
            "all_time_headpat_user": username,
            "all_time_headpat_count": q[1],
        }

    def check_best_strawberry_milk_giver(self):
        q = (
            self.session.query(
                UserRewardsRedeemed.twitch_user_id,
                func.count(UserRewardsRedeemed.twitch_user_id),
            )
            .filter(
                UserRewardsRedeemed.reward_id
                == "bce8f652-aeaa-426a-bc90-4160c9041b56"
            )
            .group_by(UserRewardsRedeemed.twitch_user_id)
            .order_by(func.count(UserRewardsRedeemed.twitch_user_id).desc())
            .first()
        )

        if not q:
            return None

        username = (
            self.session.query(User.twitch_username)
            .filter(User.twitch_user_id == q[0])
            .first()[0]
        )
        return {
            "all_time_strawberry_milk_user": username,
            "all_time_strawberry_milk_count": q[1],
        }
