# Grading boilerplate
import vm_tools as vmt
import matplotlib.pyplot as plt
import numpy as np
import copy
import PIL
import os
import io

from scipy import optimize
from wand.image import Image
import subprocess

class_list = []

def _double_gauss(x, mu1, sig1, a1, mu2, sig2, a2): 
    g1 = a1 * np.exp(-(x-mu1)**2 / (sig1**2))
    g2 = a2 * np.exp(-(x-mu2)**2 / (sig2**2))
    return g1 + g2


class Evaluation(object):
    """General class for quiz or exam"""
    def __init__(self, fdir, scanned_file, answer_file, n_questions=10, n_options=4, 
        absent_list=(), class_list_file=None):
        """Class to represent a quiz or exam. 

        Methods of this class set up grading, do the
        grading, and display results of grading.

        Parameters
        ----------
        fdir : str
            file path to directory for class. 
        scanned_file : str
            file name of scanned .pdf to load
        absent : tuple | list
            List of absent students for this quiz
        n_questions : scalar or tuple
            Number of questions to grade. If scalar, assumes one column
            of answers. If tuple, assumes len(n_questions) columns with
            number per column specified by each element of the tuple.
        n_options : scalar or tuple
            number of options per multiple choice or TF question
        """
        # Set up necessary info
        self.absent_list = absent_list
        self.answer_file = answer_file
        self.scanned_file = scanned_file        
        if class_list_file is None:
            self.class_list_file = os.path.join(fdir, 'class_list.txt')
        else:
            self.class_list_file = class_list_file
        self.n_questions = n_questions
        self.n_options = n_options
        # read test images
        self.load_class_list()
        self.read_images()
        self.load_answers()
        self.adjusted_points = None
        # fill absences
        self.fill_absences()

        self.mn = 180
        self.mx = 255
        self.mu1 = np.linspace(190, 210, 3)
        self.mu2 = np.linspace(210, 240, 3)

    def read_images(self):
        with Image(filename=self.scanned_file, resolution=200) as img:
            page_images = []
            for page_wand_image_seq in img.sequence:
                page_wand_image = Image(page_wand_image_seq)
                page_jpeg_bytes = page_wand_image.make_blob(format="jpeg")
                page_jpeg_data = io.BytesIO(page_jpeg_bytes)
                page_image = PIL.Image.open(page_jpeg_data)
                page_images.append(page_image)
        self.test_images = page_images

    def load_answers(self):
        with open(self.answer_file) as fid:
            lines = fid.readlines()
            multi_answers = [',' in x for x in lines]
            correct_answers = [x.strip().split(',') for x in lines]
        correct_answers = [tuple(c) if ma else c for c, ma in zip(correct_answers, multi_answers)]
        self.correct_answers = correct_answers
        self._get_multi_answer_key()

    def load_class_list(self):
        with open(self.class_list_file, mode='r') as fid:
            self.class_list = fid.readlines()
            self.class_list = [x.strip('\n') for x in self.class_list]

    def fill_absences(self):
        """Inserts blanks into test_images for absent students"""
        absent_indices = []
        for a in self.absent_list:
            si = self.get_student(a)
            absent_indices.append(si)

        for si in sorted(absent_indices):
            print("Inserting %d" % si)
            self.test_images.insert(si, [])

    def get_student(self, name, last=None):
        """Get index for student w/ a given first (and optionally last) name 
        """
        if isinstance(name, (list, tuple)):
            return self.get_student(*name)
        if last is None:
            return [cl.split()[0] for cl in self.class_list].index(name)
        else:
            return self.class_list.index(' '.join([name, last]))


    def check_names(self, ib=(110, 310, 800, 1600),
                    nr=None, nc=3, figsize=(16, 16)):
        """
        ib : list
            image bounds for location of name; ymin, ymax, xmin, xmax
        """
        if nr is None:
            nr = int(np.ceil(len(self.class_list) / nc))
        n_students = len(self.class_list)
        fig, axs = plt.subplots(nr, nc, figsize=figsize)
        for test, student, ax in zip(self.test_images, self.class_list, axs.flatten()):
            if test==[]:
                tt = np.zeros((ib[1]-ib[0], ib[3]-ib[2]))
            else:
                test = np.array(test)
                tt = np.array(test)[ib[0]:ib[1], ib[2]:ib[3]]
            ax.imshow(tt, cmap='gray')
            ax.set_title(student)
            ax.axis('off')
        plt.tight_layout()

    def test_question_bounds(self, top_edges, side_edges, student=None, 
        figsize=(20,20)):
        """Display grid over test answers to make sure that answers 
        will be graded correctly.
        """
        if student is None:
            idx = 0
        else:
            idx = self.get_student(student)

        if isinstance(side_edges, tuple):
            n_columns = len(side_edges)
        else:
            n_columns = 1
            side_edges = (side_edges,)
            top_edges = (top_edges,)
        print('%d columns'%n_columns)
        n_questions = self.n_questions
        n_options = self.n_options
        if not isinstance(n_questions, tuple):
            if n_columns > 1:
                n_questions = tuple([n_questions/n_columns] * n_columns)
            else:
                n_questions = (n_questions,)
        if not isinstance(n_options, tuple):
            if n_columns > 1:
                n_options = tuple([n_options] * n_columns)
            else:
                n_options = (n_options,)

        test_image = self.test_images[idx]
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(test_image, cmap='gray',)

        for te, se, nq, no in zip(top_edges, side_edges, n_questions, n_options):
            tops = np.linspace(te[0], te[1], nq + 1)
            sides = np.linspace(se[0], se[1], no + 1)

            xl = plt.xlim()
            yl = plt.ylim()
            ax.hlines(tops, *xl, color='r')
            ax.vlines(sides, *yl, color='b')
        plt.xlim(xl)
        plt.ylim(yl)
        plt.grid(axis='both')

        # Fix top / side edges for now
        self.top_edges = top_edges
        self.side_edges = side_edges

    def get_answers(self, mu1=None, mu2=None, mn=150, mx=245, 
        return_raw=False, show_bounds=False, top_edges=None, side_edges=None):
        """Gets all responses made, regardless of correct answer

        Fancy new algorithm: fits a two-bump Gaussian to the raw image values 
        read in by the code. Selects range of values that correspond to the lower
        Gaussian to be the range of values indicating an answer.

        Parameters
        ----------
        mu1 = array
            range of means to attempt for lower value, i.e. shaded-in answer,
            Guassian
        mu2 : array
            range of means to attempt for higher value, i.e. not-shaded-in answer, 
            Guassian
        """
        if side_edges is None:
            side_edges = copy.copy(self.side_edges)
        if top_edges is None:
            top_edges = copy.copy(self.top_edges)
        n_questions = copy.copy(self.n_questions)
        n_options = copy.copy(self.n_options)
        if isinstance(side_edges, tuple):
            n_columns = len(side_edges)
        else:
            n_columns = 1
            side_edges = (side_edges,)
            top_edges = (top_edges,)
        if not isinstance(n_questions, tuple):
            if n_columns > 1:
                n_questions = tuple([int(n_questions / n_columns)] * n_columns)
            else:
                n_questions = (n_questions,)
        if not isinstance(n_options, tuple):
            if n_columns > 1:
                n_options = tuple([n_options] * n_columns)
            else:
                n_options = (n_options,)
        recompute_mean_guesses_mu1 = mu1 is None
        recompute_mean_guesses_mu2 = mu2 is None
        # Preallocate
        answers = []
        raw = []
        absent = []
        # Loop over students
        for ii, test in enumerate(self.test_images):
            if test==[]:
                # Absent student
                absent.append(ii)
                continue
            # Loop over columns on answer sheet
            ans_list = []
            for te, se, nq, no in zip(top_edges, side_edges, n_questions, n_options):
                tops = np.linspace(te[0], te[1], nq + 1).astype(np.int)
                sides = np.linspace(se[0], se[1], no + 1).astype(np.int)

                t = np.array(test)
                ans = np.zeros((nq, no))
                for ir, (row_st, row_end) in enumerate(zip(tops[:-1], tops[1:])):
                    for ic, (col_st, col_end) in enumerate(zip(sides[:-1], sides[1:])):
                        ans[ir, ic] = t[row_st:row_end, col_st:col_end].mean()
                ans_list.append(ans)
            ans = np.vstack(ans_list)
            raw.append(ans)
            
        raw = np.array(raw)
        # Account for absences
        present = ~np.in1d(np.arange(len(self.test_images)), absent)
        tmp = raw.copy()
        raw = np.zeros((len(self.test_images), np.sum(n_questions), np.max(n_options)))
        raw[present] = tmp
        self.raw_answers = raw
        del tmp
        #thr = (raw[raw < bubble_thresh].max() - raw[raw < bubble_thresh].min()) * 0.5 + raw[raw < bubble_thresh].min()
        # Get all answers marked
        tfans = np.zeros(raw.shape) * np.nan
        # smoothing kernel. Look into maybe doing a KDE here.
        gg = lambda x: np.exp(-(x)**2)
        gk = gg(np.linspace(-2, 2, 7))
        gk /= np.sum(gk)
        for i, r in enumerate(raw):
            h, t = np.histogram(r[r<mx].flatten(), bins=np.arange(mn, 256))
            hh = np.convolve(h, gk, mode='same')
            tt = np.diff(t) + t[:-1]
            err = []
            params = []
            # Adjust min / max to data
            n_means = 5
            mnn = np.min(r[r<mx])
            mxx = np.max(r[r<mx])
            half_pt = mnn + (mxx - mnn) / 2
            if recompute_mean_guesses_mu1:
                mu1 = np.linspace(mnn - 5, half_pt, n_means)
            if recompute_mean_guesses_mu2:
                mu2 = np.linspace(half_pt, np.minimum(mxx + 5, mx), n_means)
            for mu1_ in mu1:
                for mu2_ in mu2: #, mu2_ in zip(mu1, mu2):
                    try:
                        pp, params_covariance = optimize.curve_fit(_double_gauss, # The executable function to be fit
                                                       tt, # The observed (here, simulated) input data to the function
                                                       hh, # The observed (here, simulated) output from the function
                                                       p0=(mu1_, 5, 1, mu2_, 5, 1), # initial guesses as to parameter values. 
                                                       bounds=([mnn, 1.5, 1, mnn, 1.5, 1], [mxx, 10, 100, mxx, 10, 100]),
                                                      )
                    except:
                        pp = [0,0,0,0,0,0]
                    pred = _double_gauss(tt, *pp)
                    mse = np.mean((pred - hh)**2)
                    err.append(mse)
                    params.append(pp)
            # Select best fit based on error
            m1, s1, a1, m2, s2, a2 = params[np.argmin(err)]
            # Define bounds based on two gaussian distributions
            # Note to self: this is getting very over-wrought
            mi = np.argmin([m1, m2])
            bounds = [[m1, m2][mi] - [s1, s2][mi]*4, np.mean([m1, m2])]
            if show_bounds:
                #print(bounds) 
                fig, ax = plt.subplots()
                ax.hist(r[r<mx], bins=np.arange(mn, 256))
                ee = np.argsort(err) / len(err)
                ax.plot(tt, _double_gauss(tt, *params[np.argmin(err)]), color='orange')
                #for pp, ee_ in zip(params, err):
                #    ax.plot(tt, _double_gauss(tt, *pp), color=plt.cm.inferno(ee_))
                yl = plt.ylim()
                ax.vlines(bounds, 0, yl[1], color='r')
                ax.set_ylim(yl)
            for j, ans in enumerate(r):
                # If no answer is given
                #if np.all(ans > no_answer_thresh):
                #    continue
                tmp = (ans > bounds[0]) & (ans < bounds[1])
                tfans[i, j] = tmp.astype(np.int)
        self.student_answers = tfans


    def show_answers(self, student, figsize=(12, 16), n_questions=None):
        """Display answers chosen by a given student
        
        """
        idx = get_student(student)

        # Lazy, this could cause bugs...
        answers = self.student_answers[idx]
        tmp, n_options = answers.shape
        if n_questions is None:
            n_questions = tmp
        side_edges = copy.copy(self.side_edges)
        top_edges = copy.copy(self.top_edges)
        if isinstance(side_edges, tuple):
            n_columns = len(side_edges)
        else:
            n_columns = 1
            side_edges = (side_edges,)
            top_edges = (top_edges,)

        if (n_columns > 1) and not isinstance(n_questions, tuple):
            n_questions = tuple([int(n_questions/n_columns)] * n_columns)

        if not isinstance(n_questions, tuple):
            n_questions = (n_questions, )

        if (n_columns > 1) and not isinstance(answers, tuple):
            qbounds = np.cumsum([0, *n_questions])
            #print(qbounds)
            answers = tuple([answers[st:fin] for st, fin in zip(qbounds[:-1], qbounds[1:])])
            #print([a.shape for a in answers])

        if not isinstance(answers, tuple):
            answers = (answers,)

        fig, ax = plt.subplots(figsize=figsize)
        test = np.array(self.test_images[idx])
        ax.imshow(test)
        xl = plt.xlim()
        yl = plt.ylim()

        for te, se, nq, anss in zip(top_edges, side_edges, n_questions, answers):
            tops = np.linspace(te[0], te[1], nq + 1)
            sides = np.linspace(se[0], se[1], n_options + 1)
            sd = np.mean(np.diff(sides))/2
            td = np.mean(np.diff(tops))/2
            for t, tfa in zip(tops, anss):
                tfa = np.nan_to_num(tfa)
                aa, = np.nonzero(tfa)
                for a in aa:
                    plt.scatter(sides[int(a)] + sd, t + td, edgecolor='c', marker='*', facecolor='None')
        plt.xlim(xl)
        plt.ylim(yl)


    def save_answers(self, student, 
                     figsize=(8.5, 11), corr_mk='O',
                     wrong_mk='X', ans_mk='^', show_test=True, fname=None, pts_bystudent=None,
                     extra_points=0, n_extra_questions=0):
        """Save graded exams to pdf files"""

        idx = self.get_student(student)

        # Lazy, this could cause bugs...
        answers = self.student_answers[idx]
        n_q, n_o = answers.shape

        if self.n_questions is None:
            n_questions = n_q
        else:
            n_questions = self.n_questions
        if self.n_options is None:
            n_options = n_o
        else: 
            n_options = self.n_options
            
        side_edges = copy.copy(self.side_edges)
        top_edges = copy.copy(self.top_edges)
        correct_answers = copy.copy(self.correct_answers)
        if isinstance(side_edges, tuple):
            n_columns = len(side_edges)
        else:
            n_columns = 1
            side_edges = (side_edges,)
            top_edges = (top_edges,)

        if not isinstance(n_questions, tuple):
            n_questions = tuple([int(n_questions / n_columns)] * n_columns)

        if not isinstance(answers, tuple):
            qbounds = np.cumsum([0, *n_questions])
            #print(qbounds)
            answers = tuple([answers[st:fin] for st, fin in zip(qbounds[:-1], qbounds[1:])])
            #print([a.shape for a in answers])
        if not isinstance(correct_answers, tuple):
            qbounds = np.cumsum([0, *n_questions])
            #print(qbounds)
            correct_answers = tuple([correct_answers[st:fin] for st, fin in zip(qbounds[:-1], qbounds[1:])])
            #print([len(a) for a in correct_answers])

        fig, ax = plt.subplots(figsize=figsize)
        test = np.array(self.test_images[idx])
        if show_test:
            ax.imshow(test)
            xl = plt.xlim()
            yl = plt.ylim()
        else:
            ymx, xmx, _ = test.shape
            xl = (-0.5, xmx-0.5)
            yl = (ymx-0.5, -0.5)
        for te, se, nq, anss, canss, in zip(top_edges, side_edges, n_questions, answers, correct_answers):
            tops = np.linspace(te[0], te[1], nq + 1)
            sides = np.linspace(se[0], se[1], n_options + 1)
            sd = np.mean(np.diff(sides))/2
            td = np.mean(np.diff(tops))/2
            for t, tfa, ctfa in zip(tops, anss, canss):
                tfa = np.nan_to_num(tfa)
                ctfa = np.nan_to_num(ctfa)
                aa, = np.nonzero(tfa)
                ca, = np.nonzero(ctfa)
                miss = set(ca) - set(aa)
                false_alarm = set(aa) - set(ca)
                #1/0
                for a in aa:
                    plt.scatter(sides[int(a)] + sd, t + td, edgecolor='k', marker=ans_mk, facecolor='None')
                for c in ca:
                    plt.text(sides[int(c)] + sd, t + td, corr_mk, color='k', fontsize=24, ha='center', va='center')
                for x in false_alarm:
                    plt.text(sides[int(x)] + sd, t + td, wrong_mk, color='k', fontsize=24, ha='center', va='center')
        plt.xlim(xl)
        plt.ylim(yl)
        ax.set_position([0,0,1,1])
        ax.axis('off')
        if pts_bystudent is not None:
            plt.text(1200,100, '%0.2f/%d'%(pts_bystudent[idx].sum() + extra_points, np.sum(n_questions) + n_extra_questions), fontsize=32)
        if fname is not None:
            fig.savefig(fname, dpi=150)
            plt.close(fig.number)

    def _concat_graded_assignments(self, fnames, out_file):
        """Concatenate graded exams or quizzes into a single pdf"""
        cmd = ['gs', '-dBATCH', '-dNOPAUSE', '-q', '-sDEVICE=pdfwrite', '-sOutputFile={out_file}'.format(out_file=out_file)]
        for fname in fnames:
            cmd += [fname]
        subprocess.call(cmd)
        for f in fnames:
            os.unlink(f)


    def _get_multi_answer_key(self):
        """Yo"""
        n_questions = copy.copy(self.n_questions)
        n_options = copy.copy(self.n_options)
        if n_options is None:
            n_options = max([max(['ABCDE'.index(a) if a not in ['T','F'] else 1 for a in ans]) for ans in answer_key]) + 1
        if isinstance(n_questions, (list, tuple)):
            n_questions = sum(n_questions)
        if isinstance(n_options, (list, tuple)):
            n_options = max(n_options)
        multi_answer_key = np.zeros((n_questions, n_options), dtype=np.int)
        multi_answer_n = np.zeros((n_questions,), dtype=np.int)
        for iquestion, ans in enumerate(self.correct_answers):
            for a in ans:
                try:
                    ii = "ABCDE".index(a)
                    n_tf = n_options
                except:
                    ii = "TF".index(a)
                    n_tf = 2
                multi_answer_key[iquestion, ii] = 1
            multi_answer_n[iquestion] = n_tf
        self.multi_answer_key = multi_answer_key
        self.multi_answer_n = multi_answer_n

    def grade_answers(self, n_options=None, multi_answer_n=None):
        """Grade exams

        Parameters
        ----------
        
        """
        # N for multi-answer questions
        if multi_answer_n is None:
            multi_answer_n = copy.copy(self.multi_answer_n)
        # Single answer questions
        single_answer_key = np.argmax(self.multi_answer_key, axis=1)
        answers = np.argmax(self.student_answers, axis=2)
        points = (answers == single_answer_key[np.newaxis, :]).astype(np.float)
        # Multi-answer questions
        tmp = (self.multi_answer_key == self.student_answers).astype(np.float)
        multi_answer_points = np.zeros_like(points)
        for i, n in enumerate(multi_answer_n):
            multi_answer_points[:, i] = tmp[:, i, :n].sum(1) / n
        multi_answer_questions = np.array([isinstance(a, tuple) for a in self.correct_answers])
        # Substitute points for multi-answer questions
        points[:, multi_answer_questions] = multi_answer_points[:, multi_answer_questions]

        # Set absent = nan
        for ab in self.absent_list:
            j = self.get_student(ab)
            points[j] = np.nan
        self.points = points


    def forgiveness_points(self, question, allowable_answers, points, reset=False):
        """Grant extra points for specific answers to specific questions

        Parameters
        ----------
        question : int
            Question number (1-based)
        allowable_answers : list or tuple
            Answers allowed. for now, should be capital letters, e.g. 
            ['A','B']
        points : scalar
            How many points to grant if a student answered one of 
            `allowable_answers` to `question`
        """ 
        # Make question into python index
        question -= 1
        if (self.adjusted_points is None) or reset:
            scores_new = self.points.copy()
        else:
            scores_new = self.adjusted_points.copy()
        allowable_answers = ['ABCDE'.index(x) for x in allowable_answers]
        answers = np.argmax(self.student_answers, axis=2)
        chose_bad_answer = np.in1d(answers[:, question], allowable_answers)
        print('Adding %0.1f points to %d students' % (points, chose_bad_answer.sum()))
        scores_new[chose_bad_answer, question] = scores_new[chose_bad_answer, question] + points
        self.adjusted_points = scores_new


    def question_metric_plot(self, test_cases=(), title=None, bad_thresh=65, ylim=None, 
                             questions_per_row=5):
        """"""
        #n_students, n_questions, n_options = raw.shape
        n_questions = copy.copy(self.n_questions)
        if isinstance(n_questions, (list, tuple)):
            n_questions = sum(n_questions)
        n_options = copy.copy(self.n_options)
        if isinstance(n_options, (list, tuple)):
            n_options = max(n_options)
        n_students = len(self.class_list)
        if ylim is None:
            ylim = (0, n_students)
        # Grade questions
        if not hasattr(self, 'points'):
            print("Please call `grade_answers()` method before calling this function")
            return
        scores_byquestion = np.nanmean(self.points, axis=0) * 100
        bad_questions = np.nonzero(scores_byquestion < bad_thresh)[0]
        
        # Set up plot
        abcde = 'ABCDE'
        n_rows = np.ceil(n_questions / questions_per_row).astype(np.int)
        bins = np.arange(-0.5, n_options - 0.4, 1)
        fig, axs = plt.subplots(n_rows, 5, figsize=(10, 2 * n_rows))
        # Loop over questions
        for q, ax in zip(range(n_questions), axs.flatten()):
            tfa_students = np.nansum(self.student_answers[:, q], 0)
            vmt.plot_utils.bar(tfa_students, xw=(np.arange(0, n_options), 0.8),
                               lw=0, ax=ax)
            if q in bad_questions:
                color = 'r'
            else:
                color = 'g'
            tfa, = np.nonzero(self.multi_answer_key[q])
            ax.vlines(tfa, *ylim, color=color, lw=5)
            ax.set_xticks(range(n_options))
            ax.set_xticklabels([abcde[i] for i in range(n_options)])
            # Test cases
            for j, tc in enumerate(test_cases):
                ti = self.get_student(tc)
                tfa_student, = np.nonzero((self.student_answers[ti, q] > 0) & ~np.isnan(self.student_answers[ti, q]))
                for sa in tfa_student:
                    ax.text(sa, j*5 + 5, tc[0], ha='center', color='m')
            ax.set_title('Question %d\n(%0.2f%% correct)' % (q+1, scores_byquestion[q]))
            ax.set_ylim(ylim)
            vmt.plot_utils.open_axes(ax)
        plt.tight_layout()
        if title is not None:
            fig.savefig(title, dpi=100)

    def score_hist(self, title=None, figsize=(6, 4)):
        """
        Scores should be students x questions"""
        n_students, n_questions = self.points.shape
        scores_ = self.points.mean(axis=1) * n_questions
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(scores_[~np.isnan(scores_)], bins=np.arange(-0.5, n_questions + 0.6, 1))
        vmt.plot_utils.open_axes(ax)
        ax.set_xlabel("Points")
        ax.set_ylabel("Students (count)")
        if title is not None:
            fig.savefig(title, dpi=100)


    def question_hist(self, title=None, figsize=(6, 4), reference_lines=(60, 75)):
        """Scores should be students x questions"""
        n_students, n_questions = self.points.shape
        scores_ = np.nanmean(self.points, axis=0) * 100
        fig, ax = plt.subplots(figsize=figsize)
        vmt.plot_utils.bar(scores_, ax=ax)
        plt.xticks(np.arange(1, n_questions + 1))
        ax.hlines(reference_lines, 0, n_questions + 1, color='r')
        vmt.plot_utils.open_axes(ax)
        ax.set_xlabel("Question")
        ax.set_ylabel("% correct")
        if title is not None:
            fig.savefig(title, dpi=100)

    def export_grades(self, file, assignment_name):
        """Insert grades (points) as a particular assignment

        Intended to work with webcampus grade exports
        """
        if '.csv' not in file:
            raise Exception('Must be a csv file')

        import pandas as pd
        df = pd.read_csv(file) # maybe need delim
        # Order grades according to index of df
        idx = []
        to_enter = points[idx]
        # insert grades as a column of df 
        df[assignment_name] = to_enter
        # Save df as new csv
        fname = file.replace('.csv', '_updated.csv')

    def list_student_scores(self):
        if self.adjusted_points is not None:
            points = self.adjusted_points
        else:
            points = self.points
        for s, p in zip(self.class_list, np.round(points.sum(1), decimals=2)):
            print('%-25s : %0.2f'%(s, p))