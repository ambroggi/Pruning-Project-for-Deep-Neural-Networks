# This is supposed to be compatibility code to transfer the model into the format expected by the code in Task-Oriented-Feature_Distillation
# The paper: https://proceedings.neurips.cc/paper_files/paper/2020/file/a96b65a721e561e1e3de768ac819ffbb-Paper.pdf
# See: https://github.com/ArchipLab-LinfengZhang/Task-Oriented-Feature-Distillation/commit/fcfd4be5ff773d2d27adccdc7df206cdf502800e
# For the origin of some code segments, code segments will be marked.
# Changes will be marked with "CHANGE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from TaskOrientedFeatureDistillation.utils import CrossEntropy
# from TaskOrientedFeatureDistillation.utils import get_orth_loss


class task_oriented_feature_wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.link = torch.nn.ModuleList()

    def forward(self):
        pass


def name_main(optimizer: torch.optim.Optimizer, teacher: task_oriented_feature_wrapper, net: task_oriented_feature_wrapper, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader, device: torch.device, LR: float, criterion: nn._Loss, args: 'ConfigCompatabilityWrapper', epochs: int = 250):
    # This is code from before the __name__=="__main__" block in /distill.py of the origin code
    init = False
    # This is code from the __name__ == "__main__" block in /distill.py of the origin code (https://github.com/ArchipLab-LinfengZhang/Task-Oriented-Feature-Distillation)
    best_acc = 0
    print("Start Training")
    for epoch in range(epochs):  # CHANGE, changed number of epochs to be a variable
        if epoch in [80, 160, 240]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, student_feature = net(inputs)

            #   get teacher results
            with torch.no_grad():
                teacher_logits, teacher_feature = teacher(inputs)

            #   init the feature resizing layer depending on the feature size of students and teachers
            #   a fully connected layer is used as feature resizing layer here
            if not init:
                teacher_feature_size = teacher_feature[0].size(1)
                student_feature_size = student_feature[0].size(1)
                num_auxiliary_classifier = len(teacher_logits)
                link = []
                for j in range(num_auxiliary_classifier):
                    link.append(nn.Linear(student_feature_size, teacher_feature_size, bias=False))
                net.link = nn.ModuleList(link)
                net.cuda()
                #   we redefine optimizer here so it can optimize the net.link layers.
                optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)
                init = True

            #   compute loss
            loss = torch.FloatTensor([0.]).to(device)

            #   Distillation Loss + Task Loss
            for index in range(len(student_feature)):
                student_feature[index] = net.link[index](student_feature[index])
                #   task-oriented feature distillation loss
                loss += torch.dist(student_feature[index], teacher_feature[index], p=2) * args.alpha
                #   task loss (cross entropy loss for the classification task)
                loss += criterion(outputs[index], labels)
                #   logit distillation loss, CrossEntropy implemented in utils.py.
                loss += CrossEntropy(outputs[index], teacher_logits[index], 1 + (args.t / 250) * float(1 + epoch))

            # Orthogonal Loss
            for index in range(len(student_feature)):
                weight = list(net.link[index].parameters())[0]
                weight_trans = weight.permute(1, 0)
                ones = torch.eye(weight.size(0)).cuda()
                ones2 = torch.eye(weight.size(1)).cuda()
                loss += torch.dist(torch.mm(weight, weight_trans), ones, p=2) * args.beta
                loss += torch.dist(torch.mm(weight_trans, weight), ones2, p=2) * args.beta

        sum_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = torch.max(outputs[0].data, 1)
        correct += float(predicted.eq(labels.data).cpu().sum())

        if i % 20 == 0:
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.2f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                     100 * correct / total))

    print("Waiting Test!")
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, feature = net(images)
            _, predicted = torch.max(outputs[0].data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())
            total += float(labels.size(0))

        # print('Test Set AccuracyAcc:  %.4f%% ' % (100 * correct / total))
        # if correct / total > best_acc:
        #     best_acc = correct / total
        #     print("Best Accuracy Updated: ", best_acc * 100)
        #     torch.save(net.state_dict(), "./checkpoint/" + args.model + ".pth")
