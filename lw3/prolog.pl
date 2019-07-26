main :-
delete_all,
find_player(Player),nl.



find_player(Player) :-
player(Player), write(Player), !.


% Удаление старых ответов
delete_all:-retract(choose(_,_)),delete_all. % переписал сейчас норм 
delete_all.





% Находит ответ из Choices
parse(0, [First|_], First).
parse(Index, [First|Rest], Response) :-
Index > 0,
NextIndex is Index - 1,
parse(NextIndex, Rest, Response).

% Вывод вопросов из Choices
answers([], _).
answers([First|Rest], Index) :-
write(Index), write(' '), answer(First), nl,
NextIndex is Index + 1,
answers(Rest, NextIndex).

% Задает и сохраняет вопрос
ask(Question, Answer, Choices) :-
question(Question),
answers(Choices, 0),
read(Index),
parse(Index, Choices, Response),
asserta(choose(Question, Response)),
Response = Answer.

 






position(Answer) :-
choose(position, Answer).
position(Answer) :-
\+ choose(position, _),
ask(position, Answer, [goalkeeper, defender, midfielder, forward]).


physics(Answer) :-
choose(physics, Answer).
physics(Answer) :-
\+ choose(physics, _),
ask(physics, Answer, [tall_strong, fast, no_matter]).


skill(Answer) :-
choose(skill, Answer).
skill(Answer) :-
\+ choose(skill, _),
ask(skill, Answer, [good_pass, defending, experience]). 


special_skills(Answer) :-
choose(special_skills, Answer).
special_skills(Answer) :-
\+ choose(special_skills, _),
ask(special_skills, Answer, [dribbler, playmaker, good_shot,no_matter2]).









%Вопросы к базе
question(position) :-
write('Position'), nl.

question(physics) :-
write('physics skills'), nl.

question(skill) :-
write('skill'), nl.

question(special_skills) :-
write('special skills'), nl.



% Ответы к position
answer(goalkeeper) :-
write('Goalkeeper').

answer(defender) :-
write('Back').

answer(midfielder) :-
write('Centre').

answer(forward) :-
write('Forward').




% Ответы к physics
answer(tall_strong) :-
write('tall, strong').

answer(fast) :-
write('fast').

answer(no_matter) :-
write('no matter').




%Ответы к skill
answer(experience) :-
write('great experience').

answer(defending) :-
write('strong defending').

answer(good_pass) :-
write('have a good pass').




%Ответы для special_skills
answer(dribbler) :-
write('dribbler').

answer(playmaker) :-
write('playmaker').

answer(good_shot) :-
write('good shot').

answer(no_matter2) :-
write('no_matter').

























































player('Kepa') :-
position(goalkeeper),
physics(tall_strong),
skill(experience),
special_skills(dribbler).

player('Filimonov') :-
position(goalkeeper),
physics(tall_strong),
skill(experience),
special_skills(playmaker).

player('Akenfeev') :-
position(goalkeeper),
physics(tall_strong),
skill(experience),
special_skills(good_shot).

player('Oblako') :-
position(goalkeeper),
physics(tall_strong),
skill(experience),
special_skills(no_matter2).

player('Apek') :-
position(goalkeeper),
physics(tall_strong),
skill(defending),
special_skills(dribbler).

player('Lewa') :-
position(goalkeeper),
physics(tall_strong),
skill(defending),
special_skills(playmaker).

player('Chmiecel') :-
position(goalkeeper),
physics(tall_strong),
skill(defending),
special_skills(good_shot).

player('Messi') :-
position(goalkeeper),
physics(tall_strong),
skill(defending),
special_skills(no_matter2).

player('Mandanda') :-
position(goalkeeper),
physics(tall_strong),
skill(good_pass),
special_skills(dribbler).

player('Erzan') :-
position(goalkeeper),
physics(tall_strong),
skill(good_pass),
special_skills(playmaker).

player('Face') :-
position(goalkeeper),
physics(tall_strong),
skill(good_pass),
special_skills(good_shot).

player('Sari') :-
position(goalkeeper),
physics(tall_strong),
skill(good_pass),
special_skills(no_matter2).


player('Isan') :-
position(goalkeeper),
physics(fast),
skill(experience),
special_skills(dribbler).

player('SixNine') :-
position(goalkeeper),
physics(fast),
skill(experience),
special_skills(playmaker).

player('Buffon') :-
position(goalkeeper),
physics(fast),
skill(experience),
special_skills(good_shot).

player('Lunev') :-
position(goalkeeper),
physics(fast),
skill(experience),
special_skills(no_matter2).

player('Adsa') :-
position(goalkeeper),
physics(fast),
skill(defending),
special_skills(dribbler).

player('Casilas') :-
position(goalkeeper),
physics(fast),
skill(defending),
special_skills(playmaker).

player('Oxxxymiron') :-
position(goalkeeper),
physics(fast),
skill(defending),
special_skills(good_shot).

player('Rucavica') :-
position(goalkeeper),
physics(fast),
skill(defending),
special_skills(no_matter2).

player('Gone Flud') :-
position(goalkeeper),
physics(fast),
skill(good_pass),
special_skills(dribbler).

player('De Gey') :-
position(goalkeeper),
physics(fast),
skill(good_pass),
special_skills(playmaker).

player('Itevangelist') :-
position(goalkeeper),
physics(fast),
skill(good_pass),
special_skills(good_shot).

player('Pkmxier') :-
position(goalkeeper),
physics(fast),
skill(good_pass),
special_skills(no_matter2).

player('Chuits') :-
position(goalkeeper),
physics(no_matter),
skill(experience),
special_skills(dribbler).

player('Malafeev') :-
position(goalkeeper),
physics(no_matter),
skill(experience),
special_skills(playmaker).

player('Buffon') :-
position(goalkeeper),
physics(no_matter),
skill(experience),
special_skills(good_shot).

player('Cech') :-
position(goalkeeper),
physics(no_matter),
skill(experience),
special_skills(no_matter2).

player('Asus') :-
position(goalkeeper),
physics(no_matter),
skill(defending),
special_skills(dribbler).

player('Russian') :-
position(goalkeeper),
physics(no_matter),
skill(defending),
special_skills(playmaker).

player('Ivanov') :-
position(goalkeeper),
physics(no_matter),
skill(defending),
special_skills(good_shot).

player('Lukin') :-
position(goalkeeper),
physics(no_matter),
skill(defending),
special_skills(no_matter2).

player('Sansund') :-
position(goalkeeper),
physics(no_matter),
skill(good_pass),
special_skills(dribbler).

player('Adidas') :-
position(goalkeeper),
physics(no_matter),
skill(good_pass),
special_skills(playmaker).

player('Petikov') :-
position(goalkeeper),
physics(no_matter),
skill(good_pass),
special_skills(good_shot).

player('Casilas') :-
position(goalkeeper),
physics(no_matter),
skill(good_pass),
special_skills(no_matter2).



















player('Ramos') :-
position(defender),
physics(tall_strong),
skill(experience),
special_skills(dribbler).

player('Varane') :-
position(defender),
physics(tall_strong),
skill(experience),
special_skills(playmaker).

player('Pique') :-
position(defender),
physics(tall_strong),
skill(experience),
special_skills(good_shot).

player('Rottenberg Borya') :-
position(defender),
physics(tall_strong),
skill(experience),
special_skills(no_matter2).

player('Trent-Shvarz') :-
position(defender),
physics(tall_strong),
skill(defending),
special_skills(dribbler).

player('Van Stena') :-
position(defender),
physics(tall_strong),
skill(defending),
special_skills(playmaker).

player('De zdjuk') :-
position(defender),
physics(tall_strong),
skill(defending),
special_skills(good_shot).

player('Carvahal') :-
position(defender),
physics(tall_strong),
skill(defending),
special_skills(no_matter2).

player('Semedo') :-
position(defender),
physics(tall_strong),
skill(good_pass),
special_skills(dribbler).

player('Alba') :-
position(defender),
physics(tall_strong),
skill(good_pass),
special_skills(playmaker).

player('Lenglet') :-
position(defender),
physics(tall_strong),
skill(good_pass),
special_skills(good_shot).

player('Luiz') :-
position(defender),
physics(tall_strong),
skill(good_pass),
special_skills(no_matter2).

player('Lonua') :-
position(defender),
physics(fast),
skill(experience),
special_skills(dribbler).

player('Facer') :-
position(defender),
physics(fast),
skill(experience),
special_skills(playmaker).

player('Risk') :-
position(defender),
physics(fast),
skill(experience),
special_skills(good_shot).

player('Kerni') :-
position(defender),
physics(fast),
skill(experience),
special_skills(no_matter2).

player('Tracsa') :-
position(defender),
physics(fast),
skill(defending),
special_skills(dribbler).

player('Velichutin') :-
position(defender),
physics(fast),
skill(defending),
special_skills(playmaker).

player('Smirnov') :-
position(defender),
physics(fast),
skill(defending),
special_skills(good_shot).

player('Ignatiev') :-
position(defender),
physics(fast),
skill(defending),
special_skills(no_matter2).

player('Rodolfo') :-
position(defender),
physics(fast),
skill(good_pass),
special_skills(dribbler).

player('Flexin') :-
position(defender),
physics(fast),
skill(good_pass),
special_skills(playmaker).

player('Gonefludd') :-
position(defender),
physics(fast),
skill(good_pass),
special_skills(good_shot).

player('Verner') :-
position(defender),
physics(fast),
skill(good_pass),
special_skills(no_matter2).

player('Wetru') :-
position(defender),
physics(no_matter),
skill(experience),
special_skills(dribbler).

player('Rudiger') :-
position(defender),
physics(no_matter),
skill(experience),
special_skills(playmaker).

player('Silva') :-
position(defender),
physics(no_matter),
skill(experience),
special_skills(good_shot).

player('Facer') :-
position(defender),
physics(no_matter),
skill(experience),
special_skills(no_matter2).

player('Skepta') :-
position(defender),
physics(no_matter),
skill(defending),
special_skills(dribbler).

player('Di canio') :-
position(defender),
physics(no_matter),
skill(defending),
special_skills(playmaker).

player('Erecsen') :-
position(defender),
physics(no_matter),
skill(defending),
special_skills(good_shot).

player('Soshnicov') :-
position(defender),
physics(no_matter),
skill(defending),
special_skills(no_matter2).

player('Uznali') :-
position(defender),
physics(no_matter),
skill(good_pass),
special_skills(dribbler).

player('Lindeloff') :-
position(defender),
physics(no_matter),
skill(good_pass),
special_skills(playmaker).

player('Zirniy') :-
position(defender),
physics(no_matter),
skill(good_pass),
special_skills(good_shot).

player('Borya') :-
position(defender),
physics(no_matter),
skill(good_pass),
special_skills(no_matter2).




















player('Valeri Zmishenko') :-
position(midfielder),
physics(tall_strong),
skill(experience),
special_skills(dribbler).

player('Flower') :-
position(midfielder),
physics(tall_strong),
skill(experience),
special_skills(playmaker).

player('Pogba') :-
position(midfielder),
physics(tall_strong),
skill(experience),
special_skills(good_shot).

player('Denisov') :-
position(midfielder),
physics(tall_strong),
skill(experience),
special_skills(no_matter2).

player('Samba') :-
position(midfielder),
physics(tall_strong),
skill(defending),
special_skills(dribbler).

player('Mac tomine') :-
position(midfielder),
physics(tall_strong),
skill(defending),
special_skills(playmaker).

player('Fred') :-
position(midfielder),
physics(tall_strong),
skill(defending),
special_skills(good_shot).

player('Neymar jr') :-
position(midfielder),
physics(tall_strong),
skill(defending),
special_skills(no_matter2).

player('Drinkwater') :-
position(midfielder),
physics(tall_strong),
skill(good_pass),
special_skills(dribbler).

player('Nasri') :-
position(midfielder),
physics(tall_strong),
skill(good_pass),
special_skills(playmaker).

player('Huionlong') :-
position(midfielder),
physics(tall_strong),
skill(good_pass),
special_skills(good_shot).

player('Son hin mi') :-
position(midfielder),
physics(tall_strong),
skill(good_pass),
special_skills(no_matter2).

player('Van der sar') :-
position(midfielder),
physics(fast),
skill(experience),
special_skills(dribbler).

player('Xawi') :-
position(midfielder),
physics(fast),
skill(experience),
special_skills(playmaker).

player('Gomes') :-
position(midfielder),
physics(fast),
skill(experience),
special_skills(good_shot).

player('Wahchi uu') :-
position(midfielder),
physics(fast),
skill(experience),
special_skills(no_matter2).

player('Valerka Borow') :-
position(midfielder),
physics(fast),
skill(defending),
special_skills(dribbler).

player('Winstown') :-
position(midfielder),
physics(fast),
skill(defending),
special_skills(playmaker).

player('Eminem') :-
position(midfielder),
physics(fast),
skill(defending),
special_skills(good_shot).

player('Reina') :-
position(midfielder),
physics(fast),
skill(defending),
special_skills(no_matter2).

player('Dzagoev') :-
position(midfielder),
physics(fast),
skill(good_pass),
special_skills(dribbler).

player('Modric') :-
position(midfielder),
physics(fast),
skill(good_pass),
special_skills(playmaker).

player('Racitic') :-
position(midfielder),
physics(fast),
skill(good_pass),
special_skills(good_shot).

player('Flately') :-
position(midfielder),
physics(fast),
skill(good_pass),
special_skills(no_matter2).

player('Vincent "Vinnie" Jones') :-
position(midfielder),
physics(no_matter),
skill(experience),
special_skills(dribbler).

player('Herderson') :-
position(midfielder),
physics(no_matter),
skill(experience),
special_skills(playmaker).

player('Wanyama') :-
position(midfielder),
physics(no_matter),
skill(experience),
special_skills(good_shot).

player('Kroos') :-
position(midfielder),
physics(no_matter),
skill(experience),
special_skills(no_matter2).

player('Kepa') :-
position(midfielder),
physics(no_matter),
skill(defending),
special_skills(dribbler).

player('Diabi') :-
position(midfielder),
physics(no_matter),
skill(defending),
special_skills(playmaker).

player('Viktor Faizulin') :-
position(midfielder),
physics(no_matter),
skill(defending),
special_skills(good_shot).

player('Nasri') :-
position(midfielder),
physics(no_matter),
skill(defending),
special_skills(no_matter2).

player('Paul Gascoigne') :-
position(midfielder),
physics(no_matter),
skill(good_pass),
special_skills(dribbler).

player('Jora') :-
position(midfielder),
physics(no_matter),
skill(good_pass),
special_skills(playmaker).

player('Dmitriy') :-
position(midfielder),
physics(no_matter),
skill(good_pass),
special_skills(good_shot).

player('Denisov') :-
position(midfielder),
physics(no_matter),
skill(good_pass),
special_skills(no_matter2).
























player('Alex Kokorin') :-
position(forward),
physics(tall_strong),
skill(experience),
special_skills(dribbler).

player('Messi') :-
position(forward),
physics(tall_strong),
skill(experience),
special_skills(playmaker).

player('Ronaldo') :-
position(forward),
physics(tall_strong),
skill(experience),
special_skills(good_shot).

player('Vagner Love') :-
position(forward),
physics(tall_strong),
skill(experience),
special_skills(no_matter2).

player('Etoo') :-
position(forward),
physics(tall_strong),
skill(defending),
special_skills(dribbler).

player('Akinfenwa') :-
position(forward),
physics(tall_strong),
skill(defending),
special_skills(playmaker).

player('Rashmond') :-
position(forward),
physics(tall_strong),
skill(defending),
special_skills(good_shot).

player('Firmina') :-
position(forward),
physics(tall_strong),
skill(defending),
special_skills(no_matter2).

player('Kepa') :-
position(forward),
physics(tall_strong),
skill(good_pass),
special_skills(dribbler).

player('Mckenli') :-
position(forward),
physics(tall_strong),
skill(good_pass),
special_skills(playmaker).

player('Anton Zabolotny') :-
position(forward),
physics(tall_strong),
skill(good_pass),
special_skills(good_shot).

player('Sychev') :-
position(forward),
physics(tall_strong),
skill(good_pass),
special_skills(no_matter2).

player('Kepa') :-
position(forward),
physics(fast),
skill(experience),
special_skills(dribbler).

player('Pogrebnyak') :-
position(forward),
physics(fast),
skill(experience),
special_skills(playmaker).

player('Kokorin') :-
position(forward),
physics(fast),
skill(experience),
special_skills(good_shot).

player('Kerzakov') :-
position(forward),
physics(fast),
skill(experience),
special_skills(no_matter2).

player('Kepa') :-
position(forward),
physics(fast),
skill(defending),
special_skills(dribbler).

player('Arshavin') :-
position(forward),
physics(fast),
skill(defending),
special_skills(playmaker).

player('Nikita Bazhenov') :-
position(forward),
physics(fast),
skill(defending),
special_skills(good_shot).

player('Ronaldo') :-
position(forward),
physics(fast),
skill(defending),
special_skills(no_matter2).

player('Scrilnikow') :-
position(forward),
physics(fast),
skill(good_pass),
special_skills(dribbler).

player('Neymar Jr') :-
position(forward),
physics(fast),
skill(good_pass),
special_skills(playmaker).

player('Mdappe') :-
position(forward),
physics(fast),
skill(good_pass),
special_skills(good_shot).

player('Mactomine') :-
position(forward),
physics(fast),
skill(good_pass),
special_skills(no_matter2).

player('Lingard') :-
position(forward),
physics(no_matter),
skill(experience),
special_skills(dribbler).

player('Wakaso') :-
position(forward),
physics(no_matter),
skill(experience),
special_skills(playmaker).

player('Hazard Jr') :-
position(forward),
physics(no_matter),
skill(experience),
special_skills(good_shot).

player('Varlamov') :-
position(forward),
physics(no_matter),
skill(experience),
special_skills(no_matter2).

player('Ovechkin') :-
position(forward),
physics(no_matter),
skill(defending),
special_skills(dribbler).

player('Wakaflakaflame') :-
position(forward),
physics(no_matter),
skill(defending),
special_skills(playmaker).

player('Winston') :-
position(forward),
physics(no_matter),
skill(defending),
special_skills(good_shot).

player('Ronaldo') :-
position(forward),
physics(no_matter),
skill(defending),
special_skills(no_matter2).

player('Kepa') :-
position(forward),
physics(no_matter),
skill(good_pass),
special_skills(dribbler).

player('Tusanbe') :-
position(forward),
physics(no_matter),
skill(good_pass),
special_skills(playmaker).

player('Paulo Di Canio') :-
position(forward),
physics(no_matter),
skill(good_pass),
special_skills(good_shot).

player('Solskaer') :-
position(forward),
physics(no_matter),
skill(good_pass),
special_skills(no_matter2).


