import random

class Game:
    def __init__(self,word,lives):
        self.word = word
        self.lives = lives
        self.guessedLetters=[]

    # 返回剩余的生命数
    def get_lives(self):
        return self.lives

    # 从当前生命值中去掉一条命
    def remove_lives(self):
        self.lives = self.lives - 1

    # 返回一个包含所有猜出字母的列表
    def get_guessed_letters(self):
        self.guessedLetters.sort()
        return self.guessedLetters

    # 将用户猜出的字母添加到列表中
    def add_guessed_letters(self, letter):
        self.guessedLetters.append(letter)

    # 这将显示单词的空格或猜出的字母
    def display_blanks(self):
        blanks = ''
        for i in self.word:
            if i in self.guessedLetters:
                blanks = blanks + i + ' '
            else:
                blanks = blanks + '_ '
        print(blanks)
    # 这将检查输入的字母是否包含在世界中
    def check(self, letter):
        if letter in self.word:
            return True
        return False
    # 这将检查用户是否通过猜测单词中的所有字母获胜
    def won(self):
        count = 0
        for i in self.word:
            if i in self.guessedLetters:
                count = count + 1
        return count == len(self.word)


# 此函数应从下面的 word_list 中随机选择六个单词之一，并return它
def get_word():
    word_list = ['business', 'reviews', 'shipping', 'december', 'provided', 'programs']
    return word_list[random.randint(0,5)]


# 这是启动游戏需要运行的唯一函数。
# 它应该在所有生命都丢失或单词被猜到时正确终止。
def start():
    # 游戏初始化
    word = get_word()
    lives = 6
    game = Game(word,lives)
    
    while True:
        # 打印当前猜测情况
        game.display_blanks()
        # 死循环终止条件 1.死亡 2.胜利
        if game.get_lives()<=0: 
            print("Sorry, you did not win the game, the word was: " + word)
            break
        if game.won():
            print("Congrats, you've won the game and guessed the word: " + word)
            break
        # 请用户输入字母、大写转小写
        guess = str(input('Please enter a letter: ')).lower()
        # 确保一次只输入一个字母
        if len(guess)>1:
            print('That is not a single letter.\n')
            continue
        # 确保字母没有被猜过
        if guess in game.get_guessed_letters():
            print('That letter has already been guessed.\n')
            continue
        # 确保输入的是字母而不是其他字符
        if guess<'a' or guess>'z':
            print('That is not a single letter.\n')
            continue
        # 加入所猜的数字，若猜错则扣血
        game.add_guessed_letters(guess)
        if not game.check(guess):
            game.remove_lives()
        print('Guessed letters:',game.get_guessed_letters(),'Lives remaining:',game.get_lives(),'\n')
        


if __name__=='__main__':
    start()