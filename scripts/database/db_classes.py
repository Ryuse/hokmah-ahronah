import logging

from sqlalchemy import create_engine, ForeignKey, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

#This disables logging from sqlalchemy
logging.basicConfig()
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)

Base = declarative_base()

class User(Base):
	__tablename__ = "twitch_users"
	
	twitch_user_id = Column("twitch_user_id", Integer, primary_key=True)
	twitch_username = Column("twitch_username", String)
	twitch_past_username = Column("twitch_past_username", String, default="")
	time_created = Column("time_created", DateTime(timezone=True), server_default=func.now())
	time_updated = Column("time_updated", DateTime(timezone=True), onupdate=func.now())
	
	def __init__(self, twitch_user_id, twitch_username):
		self.twitch_user_id = twitch_user_id
		self.twitch_username = twitch_username
		#self.twitch_past_username = twitch_past_username
		#self.time_created = time_created
		#self.time_updated = time_updated
		
		
	def __repr__(self):
		return f"""twitch_user_id: {self.twitch_user_id}
		twitch_username: {self.twitch_username}
		twitch_past_username: {self.twitch_past_username}
		time_created: {self.time_created}
		time_updated: {self.time_updated}"""

class Message(Base):
	__tablename__ = "twitch_messages"
	
	message_id = Column("message_id", Integer, primary_key = True)
	twitch_user_id = Column("twitch_user_id", Integer, ForeignKey("twitch_users.twitch_user_id"))


	message_text = Column("message_text", String)
	time_created = Column("time_created", DateTime(timezone=True), server_default=func.now())
	time_updated = Column("time_updated", DateTime(timezone=True), onupdate=func.now())
	
	def __init__(self, twitch_user_id, message_text):
		self.twitch_user_id = twitch_user_id
		self.message_text = message_text
		#self.time_created = time_created
		#self.time_updated = time_updated
		
		
	def __repr__(self):
		return f"""twitch_user_id: {self.twitch_user_id}
		message_text: {self.message_text}
		time_created: {self.time_created}
		time_updated: {self.time_updated}"""

class UnknownMessage(Base):
	__tablename__ = "twitch_unknown_messages"
	
	message_id = Column("message_id", Integer, primary_key = True)
	twitch_user_id = Column("twitch_user_id", Integer, ForeignKey("twitch_users.twitch_user_id"))


	unknown_message_text = Column("unknown_message_text", String)
	time_created = Column("time_created", DateTime(timezone=True), server_default=func.now())
	time_updated = Column("time_updated", DateTime(timezone=True), onupdate=func.now())
	
	def __init__(self, twitch_user_id, unknown_message_text):
		self.twitch_user_id = twitch_user_id
		self.unknown_message_text = unknown_message_text
		#self.time_created = time_created
		#self.time_updated = time_updated
		
		
	def __repr__(self):
		return f"""twitch_user_id: {self.twitch_user_id}
		unknown_message_text: {self.unknown_message_text}
		time_created: {self.time_created}
		time_updated: {self.time_updated}"""

class UserMessageContext(Base):
	__tablename__ = "twitch_user_message_contexts"
	
	context_id = Column("context_id", Integer, primary_key = True)
	twitch_user_id = Column("twitch_user_id", Integer, ForeignKey("twitch_users.twitch_user_id"))
	message_context = Column("message_context", String)
	time_created = Column("time_created", DateTime(timezone=True), server_default=func.now())
	time_updated = Column("time_updated", DateTime(timezone=True), onupdate=func.now())
	
	def __init__(self, twitch_user_id, message_context):
		self.twitch_user_id = twitch_user_id
		self.message_context = message_context
		#self.time_created = time_created
		#self.time_updated = time_updated
		
		
	def __repr__(self):
		return f"""twitch_user_id: {self.twitch_user_id}
		message_context: {self.message_context}
		time_created: {self.time_created}
		time_updated: {self.time_updated}"""


class UserRewardsRedeemed(Base):
	__tablename__ = "twitch_user_rewards_redeemed"

	redeem_id = Column("redeem_id", Integer, primary_key=True)
	twitch_user_id = Column("twitch_user_id", Integer, ForeignKey("twitch_users.twitch_user_id"))
	reward_id = Column("reward_id", String)
	time_created = Column("time_created", DateTime(timezone=True), server_default=func.now())
	time_updated = Column("time_updated", DateTime(timezone=True), onupdate=func.now())

	def __init__(self, twitch_user_id, reward_id):
		self.twitch_user_id = twitch_user_id
		self.reward_id = reward_id

	# self.time_created = time_created
	# self.time_updated = time_updated

	def __repr__(self):
		return f"""twitch_user_id: {self.twitch_user_id}
		reward_id: {self.reward_id}
		time_created: {self.time_created}
		time_updated: {self.time_updated}"""


def main():
	engine = create_engine("sqlite:///twitch.db", echo=True)

	Base.metadata.create_all(bind=engine)

	Session = sessionmaker(bind=engine)
	session = Session()

	#user = User(852761798, "hokmahahronah")

	#msg = Message(852761798, "Wat")

	#session.add(user)
	#session.add(msg)
	#session.commit()
	#
	# results = session.query(User).all()
	# r2 = session.query(Message).all()
	# print(results)
	# print(r2)

	session.close()

if __name__ == '__main__':
	main()

